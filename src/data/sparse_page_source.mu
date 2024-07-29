/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include "../common/device_helpers.muh"  // for CurrentDevice
#include "proxy_dmatrix.muh"             // for Dispatch, DMatrixProxy
#include "simple_dmatrix.muh"            // for CopyToSparsePage
#include "sparse_page_source.h"
#include "xgboost/data.h"  // for SparsePage

namespace xgboost::data {
namespace detail {
std::size_t NSamplesDevice(DMatrixProxy *proxy) {
  return musa_impl::Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
}

std::size_t NFeaturesDevice(DMatrixProxy *proxy) {
  return musa_impl::Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
}
}  // namespace detail

void DevicePush(DMatrixProxy *proxy, float missing, SparsePage *page) {
  auto device = proxy->Device();
  if (!device.IsMUSA()) {
    device = DeviceOrd::MUSA(dh::CurrentDevice());
  }
  CHECK(device.IsMUSA());

  musa_impl::Dispatch(proxy,
                      [&](auto const &value) { CopyToSparsePage(value, device, missing, page); });
}
}  // namespace xgboost::data
