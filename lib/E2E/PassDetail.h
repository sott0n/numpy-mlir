#ifndef E2E_PASSDETAIL_H
#define E2E_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Dialect/TCP/IR/TCPDialect.h"

namespace mlir {
class ModuleOp;

namespace npc {

#define GEN_PASS_CLASSES
#include "E2E/Passes.h.inc"

} // namespace npc
} // namespace mlir

#endif // E2E_PASSDETAIL_H
