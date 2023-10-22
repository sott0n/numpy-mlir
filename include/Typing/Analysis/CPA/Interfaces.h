#ifndef NPC_TYPING_ANALYSIS_CPA_INTERFACES_H
#define NPC_TYPING_ANALYSIS_CPA_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "Typing/Analysis/CPA/Types.h"

namespace mlir {

#include "Typing/Analysis/CPA/OpInterfaces.h.inc"
#include "Typing/Analysis/CPA/TypeInterfaces.h.inc"

} // namespace mlir

#endif // NPC_TYPING_ANALYSIS_CPA_INTERFACES_H
