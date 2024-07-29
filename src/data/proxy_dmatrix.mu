/**
 * Copyright 2020-2023, XGBoost contributors
 */
#include "device_adapter.muh"
#include "proxy_dmatrix.muh"
#include "proxy_dmatrix.h"

namespace xgboost::data {
void DMatrixProxy::FromMUSAColumnar(StringView interface_str) {
  auto adapter{std::make_shared<CudfAdapter>(interface_str)};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (!adapter->Device().IsMUSA()) {
    // empty data
    CHECK_EQ(this->Info().num_row_, 0);
    ctx_ = ctx_.MakeMUSA(dh::CurrentDevice());
    return;
  }
  ctx_ = ctx_.MakeMUSA(adapter->Device().ordinal);
}

void DMatrixProxy::FromMUSAArray(StringView interface_str) {
  auto adapter(std::make_shared<CupyAdapter>(StringView{interface_str}));
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (!adapter->Device().IsMUSA()) {
    // empty data
    CHECK_EQ(this->Info().num_row_, 0);
    ctx_ = ctx_.MakeMUSA(dh::CurrentDevice());
    return;
  }
  ctx_ = ctx_.MakeMUSA(adapter->Device().ordinal);
}

namespace musa_impl {
std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const* ctx,
                                                std::shared_ptr<DMatrixProxy> proxy,
                                                float missing) {
  return Dispatch<false>(proxy.get(), [&](auto const& adapter) {
    auto p_fmat = std::shared_ptr<DMatrix>{DMatrix::Create(adapter.get(), missing, ctx->Threads())};
    return p_fmat;
  });
}
}  // namespace cuda_impl
}  // namespace xgboost::data
