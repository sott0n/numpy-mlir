#ifndef NPC_TYPING_TRANSFORMS_PASSDETAIL_H
#define NPC_TYPING_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace npc {
namespace Typing {

#define GEN_PASS_CLASSES
#include "Typing/Transforms/Passes.h.inc"

} // namespace Typing
} // namespace npc
} // namespace mlir

#endif // NPC_TYPING_TRANSFORMS_PASSDETAIL_H
