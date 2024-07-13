/*!
 * Copyright 2018-2022 XGBoost contributors
 */
#include "common.h"

namespace xgboost {
namespace common {

void SetDevice(std::int32_t device) {
  if (device >= 0) {
    dh::safe_cuda(musaSetDevice(device));
  }
}

int AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // musaGetDeviceCount will fail.
    dh::safe_cuda(musaGetDeviceCount(&n_visgpus));
  } catch (const dmlc::Error &) {
    musaGetLastError();  // reset error.
    return 0;
  }
  return n_visgpus;
}

}  // namespace common
}  // namespace xgboost
