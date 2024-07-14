/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <thrust/functional.h>                  // for maximum
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/logical.h>                     // for none_of, all_of
#include <thrust/pair.h>                        // for pair, make_pair
#include <thrust/reduce.h>                      // for reduce
#include <thrust/scan.h>                        // for inclusive_scan

#include <cstddef>                              // for size_t

#include "algorithm.muh"                        // for SegmentedArgSort
#include "musa_context.muh"                     // for CUDAContext
#include "device_helpers.muh"                   // for MakeTransformIterator, LaunchN
#include "optional_weight.h"                    // for MakeOptionalWeights, OptionalWeights
#include "ranking_utils.muh"                    // for ThreadsForMean
#include "ranking_utils.h"
#include "threading_utils.muh"                  // for SegmentedTrapezoidThreads
#include "xgboost/base.h"                       // for XGBOOST_DEVICE, bst_group_t
#include "xgboost/context.h"                    // for Context
#include "xgboost/linalg.h"                     // for VectorView, All, Range
#include "xgboost/logging.h"                    // for CHECK
#include "xgboost/span.h"                       // for Span

namespace xgboost::ltr {
namespace musa_impl {
void CalcQueriesDCG(Context const* ctx, linalg::VectorView<float const> d_labels,
                    common::Span<std::size_t const> d_sorted_idx, bool exp_gain,
                    common::Span<bst_group_t const> d_group_ptr, std::size_t k,
                    linalg::VectorView<double> out_dcg) {
  CHECK_EQ(d_group_ptr.size() - 1, out_dcg.Size());
  using IdxGroup = thrust::pair<std::size_t, std::size_t>;
  auto group_it = dh::MakeTransformIterator<IdxGroup>(
      thrust::make_counting_iterator(0ull), [=] XGBOOST_DEVICE(std::size_t idx) {
        return thrust::make_pair(idx, dh::SegmentId(d_group_ptr, idx));  // NOLINT
      });
  auto value_it = dh::MakeTransformIterator<double>(
      group_it,
      [exp_gain, d_labels, d_group_ptr, k,
       d_sorted_idx] XGBOOST_DEVICE(IdxGroup const& l) -> double {
        auto g_begin = d_group_ptr[l.second];
        auto g_size = d_group_ptr[l.second + 1] - g_begin;

        auto idx_in_group = l.first - g_begin;
        if (idx_in_group >= k) {
          return 0.0;
        }
        double gain{0.0};
        auto g_sorted_idx = d_sorted_idx.subspan(g_begin, g_size);
        auto g_labels = d_labels.Slice(linalg::Range(g_begin, g_begin + g_size));

        if (exp_gain) {
          gain = ltr::CalcDCGGain(g_labels(g_sorted_idx[idx_in_group]));
        } else {
          gain = g_labels(g_sorted_idx[idx_in_group]);
        }
        double discount = CalcDCGDiscount(idx_in_group);
        return gain * discount;
      });

  CHECK(out_dcg.Contiguous());
  std::size_t bytes;
  cub::DeviceSegmentedReduce::Sum(nullptr, bytes, value_it, out_dcg.Values().data(),
                                  d_group_ptr.size() - 1, d_group_ptr.data(),
                                  d_group_ptr.data() + 1, ctx->MUSACtx()->Stream());
  dh::TemporaryArray<char> temp(bytes);
  cub::DeviceSegmentedReduce::Sum(temp.data().get(), bytes, value_it, out_dcg.Values().data(),
                                  d_group_ptr.size() - 1, d_group_ptr.data(),
                                  d_group_ptr.data() + 1, ctx->MUSACtx()->Stream());
}

void CalcQueriesInvIDCG(Context const* ctx, linalg::VectorView<float const> d_labels,
                        common::Span<bst_group_t const> d_group_ptr,
                        linalg::VectorView<double> out_inv_IDCG, ltr::LambdaRankParam const& p) {
  CHECK_GE(d_group_ptr.size(), 2ul);
  size_t n_groups = d_group_ptr.size() - 1;
  CHECK_EQ(out_inv_IDCG.Size(), n_groups);
  dh::device_vector<std::size_t> sorted_idx(d_labels.Size());
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  common::SegmentedArgSort<false, true>(ctx, d_labels.Values(), d_group_ptr, d_sorted_idx);
  CalcQueriesDCG(ctx, d_labels, d_sorted_idx, p.ndcg_exp_gain, d_group_ptr, p.TopK(), out_inv_IDCG);
  dh::LaunchN(out_inv_IDCG.Size(), ctx->MUSACtx()->Stream(),
              [out_inv_IDCG] XGBOOST_DEVICE(size_t idx) mutable {
                double idcg = out_inv_IDCG(idx);
                out_inv_IDCG(idx) = CalcInvIDCG(idcg);
              });
}
}  // namespace cuda_impl

namespace {
struct CheckNDCGOp {
  MUSAContext const* cuctx;
  template <typename It, typename Op>
  bool operator()(It beg, It end, Op op) {
    return thrust::none_of(cuctx->CTP(), beg, end, op);
  }
};
struct CheckMAPOp {
  MUSAContext const* cuctx;
  template <typename It, typename Op>
  bool operator()(It beg, It end, Op op) {
    return thrust::all_of(cuctx->CTP(), beg, end, op);
  }
};

struct ThreadGroupOp {
  common::Span<bst_group_t const> d_group_ptr;
  std::size_t n_pairs;

  common::Span<std::size_t> out_thread_group_ptr;

  XGBOOST_DEVICE void operator()(std::size_t i) {
    out_thread_group_ptr[i + 1] =
        musa_impl::ThreadsForMean(d_group_ptr[i + 1] - d_group_ptr[i], n_pairs);
  }
};

struct GroupSizeOp {
  common::Span<bst_group_t const> d_group_ptr;

  XGBOOST_DEVICE auto operator()(std::size_t i) -> std::size_t {
    return d_group_ptr[i + 1] - d_group_ptr[i];
  }
};

struct WeightOp {
  common::OptionalWeights d_weight;
  XGBOOST_DEVICE auto operator()(std::size_t i) -> double { return d_weight[i]; }
};
}  // anonymous namespace

void RankingCache::InitOnMUSA(Context const* ctx, MetaInfo const& info) {
  MUSAContext const* cuctx = ctx->MUSACtx();

  group_ptr_.SetDevice(ctx->Device());
  if (info.group_ptr_.empty()) {
    group_ptr_.Resize(2, 0);
    group_ptr_.HostVector()[1] = info.num_row_;
  } else {
    auto const& h_group_ptr = info.group_ptr_;
    group_ptr_.Resize(h_group_ptr.size());
    auto d_group_ptr = group_ptr_.DeviceSpan();
    dh::safe_cuda(musaMemcpyAsync(d_group_ptr.data(), h_group_ptr.data(), d_group_ptr.size_bytes(),
                                  musaMemcpyHostToDevice, cuctx->Stream()));
  }

  auto d_group_ptr = DataGroupPtr(ctx);
  std::size_t n_groups = Groups();

  auto it = dh::MakeTransformIterator<std::size_t>(thrust::make_counting_iterator(0ul),
                                                   GroupSizeOp{d_group_ptr});
  max_group_size_ =
      thrust::reduce(cuctx->CTP(), it, it + n_groups, 0ul, thrust::maximum<std::size_t>{});

  threads_group_ptr_.SetDevice(ctx->Device());
  threads_group_ptr_.Resize(n_groups + 1, 0);
  auto d_threads_group_ptr = threads_group_ptr_.DeviceSpan();
  if (param_.HasTruncation()) {
    n_musa_threads_ =
        common::SegmentedTrapezoidThreads(d_group_ptr, d_threads_group_ptr, Param().NumPair());
  } else {
    auto n_pairs = Param().NumPair();
    dh::LaunchN(n_groups, cuctx->Stream(),
                ThreadGroupOp{d_group_ptr, n_pairs, d_threads_group_ptr});
    thrust::inclusive_scan(cuctx->CTP(), dh::tcbegin(d_threads_group_ptr),
                           dh::tcend(d_threads_group_ptr), dh::tbegin(d_threads_group_ptr));
    n_musa_threads_ = info.num_row_ * param_.NumPair();
  }

  sorted_idx_cache_.SetDevice(ctx->Device());
  sorted_idx_cache_.Resize(info.labels.Size(), 0);

  auto weight = common::MakeOptionalWeights(ctx, info.weights_);
  auto w_it =
      dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul), WeightOp{weight});
  weight_norm_ = static_cast<double>(n_groups) / thrust::reduce(w_it, w_it + n_groups);
}

common::Span<std::size_t const> RankingCache::MakeRankOnMUSA(Context const* ctx,
                                                             common::Span<float const> predt) {
  auto d_sorted_idx = sorted_idx_cache_.DeviceSpan();
  auto d_group_ptr = DataGroupPtr(ctx);
  common::SegmentedArgSort<false, true>(ctx, predt, d_group_ptr, d_sorted_idx);
  return d_sorted_idx;
}

void NDCGCache::InitOnMUSA(Context const* ctx, MetaInfo const& info) {
  MUSAContext const* cuctx = ctx->MUSACtx();
  auto labels = info.labels.View(ctx->Device()).Slice(linalg::All(), 0);
  CheckNDCGLabels(this->Param(), labels, CheckNDCGOp{cuctx});

  auto d_group_ptr = this->DataGroupPtr(ctx);

  std::size_t n_groups = d_group_ptr.size() - 1;
  inv_idcg_ = linalg::Zeros<double>(ctx, n_groups);
  auto d_inv_idcg = inv_idcg_.View(ctx->Device());
  musa_impl::CalcQueriesInvIDCG(ctx, labels, d_group_ptr, d_inv_idcg, this->Param());
  CHECK_GE(this->Param().NumPair(), 1ul);

  discounts_.SetDevice(ctx->Device());
  discounts_.Resize(MaxGroupSize());
  auto d_discount = discounts_.DeviceSpan();
  dh::LaunchN(MaxGroupSize(), cuctx->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) { d_discount[i] = CalcDCGDiscount(i); });
}

void PreCache::InitOnMUSA(Context const* ctx, MetaInfo const& info) {
  auto const d_label = info.labels.View(ctx->Device()).Slice(linalg::All(), 0);
  CheckPreLabels("pre", d_label, CheckMAPOp{ctx->MUSACtx()});
}

void MAPCache::InitOnMUSA(Context const* ctx, MetaInfo const& info) {
  auto const d_label = info.labels.View(ctx->Device()).Slice(linalg::All(), 0);
  CheckPreLabels("map", d_label, CheckMAPOp{ctx->MUSACtx()});
}
}  // namespace xgboost::ltr
