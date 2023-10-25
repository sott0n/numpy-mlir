#ifndef NPC_DIALECT_TCF_TRANSFORMS_PASSDETAIL_H
#define NPC_DIALECT_TCF_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "Dialect/TCF/Transforms/Passes.h"

namespace mlir {
namespace npc {
namespace tcf {

#define GEN_PASS_CLASSES
#include "Dialect/TCF/Transforms/Passes.h.inc"

} // namespace tcf
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_TCF_TRANSFORMS_PASSDETAIL_H
