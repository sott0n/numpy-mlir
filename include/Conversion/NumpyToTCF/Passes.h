#ifndef NPC_CONVERSION_NUMPYTOTCF_PASSES_H
#define NPC_CONVERSION_NUMPYTOTCF_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Dialect/Numpy/IR/NumpyDialect.h"
#include <memory>

namespace mlir {
namespace npc {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertNumpyToTCFPass();

} // namespace npc
} // namespace mlir

#endif // NPC_CONVERSION_NUMPYTOTCF_PASSES_H