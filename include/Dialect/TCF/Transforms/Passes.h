#ifndef NPC_DIALECT_TCF_TRANSFORMS_PASSES_H
#define NPC_DIALECT_TCF_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace npc {
namespace tcf {

std::unique_ptr<OperationPass<func::FuncOp>> createShapeRefinementPass();

} // namespace tcf
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_TCF_TRANSFORMS_PASSES_H