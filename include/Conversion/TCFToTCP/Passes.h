#ifndef NPC_CONVERSION_TCFTOTCP_CONVERTTCFTOTCP_H
#define NPC_CONVERSION_TCFTOTCP_CONVERTTCFTOTCP_H

#include "mlir/Pass/Pass.h"
#include "Dialect/TCF/IR/TCFDialect.h"
#include "Dialect/TCP/IR/TCPDialect.h"
#include <memory>

namespace mlir {
class ModuleOp;

namespace npc {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTCFToTCPPass();

} // namespace npc
} // namespace mlir

#endif // NPC_CONVERSION_TCFTOTCP_CONVERTTCFTOTCP_H
