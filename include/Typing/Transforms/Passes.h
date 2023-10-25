#ifndef NPC_TYPEING_TRANSFORMS_PASSES_H
#define NPC_TYPEING_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace npc {
namespace Typing {

std::unique_ptr<OperationPass<func::FuncOp>>
createCPAFunctionTypeInferencePass();

} // namespace Typing
} // namespace npc
} // namespace mlir

#endif // NPC_TYPEING_TRANSFORMS_PASSES_H
