#include "Dialect/TCP/IR/TCPDialect.h"
#include "Dialect/TCP/IR/TCPOps.h"

#include "Dialect/TCP/IR/TCPOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::npc::tcp;

void TCPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/TCP/IR/TCPOps.cpp.inc"
      >();
}
