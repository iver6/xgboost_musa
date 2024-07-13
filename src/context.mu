/**
 * Copyright 2022 by XGBoost Contributors
 */
#include "common/musa_context.muh"  // MUSAContext
#include "xgboost/context.h"


//TODO:cuctx_在context.h中定义。改为muctx_，需要在context.h增加MUSA相关的定义。
namespace xgboost {
MUSAContext const* Context::MUSACtx() const {
  if (!muctx_) {
    muctx_.reset(new MUSAContext{});
  }
  return muctx_.get();
}
}  // namespace xgboost
