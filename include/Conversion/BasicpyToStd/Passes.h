#ifndef NPC_CONVERSION_BASICPYTOSTD_PASSES_H
#define NPC_CONVERSION_BASICPYTOSTD_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include <memory>

namespace mlir {
namespace npc {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertBasicpyToStdPass();

} // namespace npc
} // namespace mlir

#endif // NPC_CONVERSION_BASICPYTOSTD_PASSES_H
