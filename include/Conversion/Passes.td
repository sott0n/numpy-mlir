#ifndef NPC_CONVERSION_PASSES
#define NPC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// Basicpy conversions
//===----------------------------------------------------------------------===//

def ConvertBasicpyToStd : Pass<"convert-basicpy-to-std", "func::FuncOp"> {
  let summary = "Convert representable Basicpy ops to std";
  let constructor = "mlir::npc::createConvertBasicpyToStdPass()";
  let dependentDialects = ["Basicpy::BasicpyDialect", "arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// Numpy conversions
//===----------------------------------------------------------------------===//

def ConvertNumpyToTCF : Pass<"convert-numpy-to-tcf", "func::FuncOp"> {
  let summary = "Convert the numpy dialect to supported TCF ops";
  let constructor = "mlir::npc::createConvertNumpyToTCFPass()";
  let dependentDialects = ["Numpy::NumpyDialect"];
}

////===----------------------------------------------------------------------===//
//// TCF conversions
////===----------------------------------------------------------------------===//
//
//def ConvertTCFToTCP : Pass<"convert-tcf-to-tcp", "ModuleOp"> {
//  let summary = "Convert TCF to TCP";
//  let constructor = "mlir::il::createConvertTCFToTCPPass()";
//}
//
////===----------------------------------------------------------------------===//
//// TCP conversions
////===----------------------------------------------------------------------===//
//
//def ConvertTCPToLinalg : Pass<"convert-tcp-to-linalg", "ModuleOp"> {
//  let summary = "Convert TCP to Linalg";
//  let constructor = "mlir::il::createConvertTCPToLinalgPass()";
//}

#endif // NPC_CONVERSION_PASSES