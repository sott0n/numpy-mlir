#ifndef NPC_DIALECT_BASICPY_IR_BASICPY_OPS_H
#define NPC_DIALECT_BASICPY_IR_BASICPY_OPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Dialect/Basicpy/IR/BasicpyOpsEnums.h.inc"
#include "Dialect/Basicpy/IR/BasicpyDialect.h"

#define GET_OP_CLASSES
#include "Dialect/Basicpy/IR/BasicpyOps.h.inc"

#endif // NPC_DIALECT_BASICPY_IR_BASICPY_OPS_H
