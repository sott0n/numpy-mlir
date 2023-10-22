#ifndef NPC_DIALECT_NUMPY_TRANSFORMS_PASSDETAIL_H
#define NPC_DIALECT_NUMPY_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "Dialect/Numpy/Transforms/Passes.h"

namespace mlir {
namespace npc {
namespace Numpy {

#define GEN_PASS_CLASSES
#include "Dialect/Numpy/Transforms/Passes.h.inc"

} // namespace Numpy
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_NUMPY_TRANSFORMS_PASSDETAIL_H