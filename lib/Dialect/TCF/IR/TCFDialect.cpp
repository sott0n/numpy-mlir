#include "Dialect/TCF/IR/TCFDialect.h"
#include "Dialect/TCF/IR/TCFOps.h"

#include "Dialect/TCF/IR/TCFOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::npc::tcf;

void TCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/TCF/IR/TCFOps.cpp.inc"
      >();
}
