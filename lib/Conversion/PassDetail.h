#ifndef NPC_CONVERSION_PASSDETAIL_H
#define NPC_CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Conversion/BasicpyToStd/Passes.h"
#include "Conversion/NumpyToTCF/Passes.h"
#include "Conversion/TCFToTCP/Passes.h"
//#include "Conversion/TCPToLinalg/Passes.h"

namespace mlir {
namespace npc {

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // namespace npc
} // namespace mlir

#endif // NPC_CONVERSION_PASSDETAIL_H
