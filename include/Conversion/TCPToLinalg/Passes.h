#ifndef NPC_CONVERSION_TCPTOLINALG_PASSES_H
#define NPC_CONVERSION_TCPTOLINALG_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <memory>

namespace mlir {
class ModuleOp;

namespace npc {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTCPToLinalgPass();

} // namespace npc
} // namespace mlir

#endif // NPC_CONVERSION_TCPTOLINALG_PASSES_H
