#ifndef NPC_BASICPY_PASSES
#define NPC_BASICPY_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def FunctionTypeInference : Pass<"basicpy-type-inference", "func::FuncOp"> {
  let summary = "Performs function level type inference";
  let constructor = "mlir::npc::Basicpy::createFunctionTypeInferencePass()";
}

#endif // NPC_BASICPY_PASSES
