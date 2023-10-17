#ifndef NPC_INITIAL_H
#define NPC_INITIAL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace npc {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace npc
} // namespace mlir

#endif // NPC_INITIAL_H