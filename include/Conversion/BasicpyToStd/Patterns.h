#ifndef NPC_CONVERSION_BASICPYTOSTD_PATTERNS_H
#define NPC_CONVERSION_BASICPYTOSTD_PATTERNS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {
namespace npc {

void populateBasicpyToStdPrimitiveOpPatterns(MLIRContext *context,
                                             RewritePatternSet &patterns);

} // namespace npc
} // namespace mlir

#endif // IL_CONVERSION_BASICPYTOSTD_PATTERNS_H
