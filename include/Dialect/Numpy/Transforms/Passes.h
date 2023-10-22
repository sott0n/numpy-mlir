#ifndef NPC_DIALECT_NUMPY_TRANSFORMS_PASSES_H
#define NPC_DIALECT_NUMPY_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace npc {
namespace Numpy {

std::unique_ptr<OperationPass<ModuleOp>> createPublicFunctionsToTensorPass();

} // namespace Numpy
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_NUMPY_TRANSFORMS_PASSES_H