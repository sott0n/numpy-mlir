#ifndef NPC_DIALECT_NPCRT_IR_NPCRTOPS_H
#define NPC_DIALECT_NPCRT_IR_NPCRTOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "Dialect/Npcrt/IR/NpcrtOps.h.inc"

#endif // NPC_DIALECT_NPCRT_IR_NPCRTOPS_H
