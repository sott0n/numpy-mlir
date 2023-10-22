//===- np-opt.cpp - The numpy compiler-----------------------------------===//
//
// This file executes as entry point of numpy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "InitAll.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::npc::registerAllPasses();

  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::npc::registerAllDialects(registry);

  return asMainReturnCode(MlirOptMain(argc, argv, "Numpy Compiler Opt Tool", registry));
}
