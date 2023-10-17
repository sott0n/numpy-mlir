#ifndef NPC_DIALECT_BASICPY_TRANSFORMS_PASSES_H
#define NPC_DIALECT_BASICPY_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <memory>

namespace mlir {
namespace npc {
namespace Basicpy {

std::unique_ptr<OperationPass<func::FuncOp>> createFunctionTypeInferencePass();

} // namespace Basicpy
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_BASICPY_TRANSFORMS_PASSES_H
