#ifndef NPC_DIALECT_BASICPY_TRANSFORMS_PASSDETAIL_H
#define NPC_DIALECT_BASICPY_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace npc {
namespace Basicpy {

#define GEN_PASS_CLASSES
#include "Dialect/Basicpy/Transforms/Passes.h.inc"

} // namespace Basicpy
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_BASICPY_TRANSFORMS_PASSDETAIL_H