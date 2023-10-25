#ifndef NPC_DIALECT_TCP_IR_TCPOPS_H
#define NPC_DIALECT_TCP_IR_TCPOPS_H

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/TCP/IR/TCPOps.h.inc"

#endif // NPC_DIALECT_TCP_IR_TCPOPS_H
