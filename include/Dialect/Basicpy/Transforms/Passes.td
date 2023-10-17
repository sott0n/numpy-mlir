#ifndef IL_BASICPY_PASSES
#define IL_BASICPY_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def FunctionTypeInference : Pass<"basicpy-type-inference", "func::FuncOp"> {
  let summary = "Performs function level type inference";
  let constructor = "mlir::il::Basicpy::createFunctionTypeInferencePass()";
}

#endif // IL_BASICPY_PASSES
