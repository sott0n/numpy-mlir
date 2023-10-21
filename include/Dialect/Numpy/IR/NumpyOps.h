#ifndef NPC_DIALECT_NUMPY_IR_NUMPY_OPS_H
#define NPC_DIALECT_NUMPY_IR_NUMPY_OPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

//#include "Typing/Analysis/CPA/Interfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Numpy/IR/NumpyOps.h.inc"

#endif // NPC_DIALECT_NUMPY_IR_NUMPY_OPS_H