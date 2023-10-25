#ifndef NPC_TYPING_TRANSFORMS_PASSES
#define NPC_TYPING_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def CPAFunctionTypeInference : Pass<"npc-cpa-type-inference", "func::FuncOp"> {
  let summary = "Performs CPA function level type inference";
  let constructor = "mlir::npc::Typing::createCPAFunctionTypeInferencePass()";
}

#endif // NPC_TYPING_TRANSFORMS_PASSES
