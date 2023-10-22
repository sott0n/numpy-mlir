#ifndef NPC_NUMPY_PASSES
#define NPC_NUMPY_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def NumpyPublicFunctionsToTensor : Pass<"numpy-public-functions-to-tensor", "ModuleOp"> {
  let summary = "Converts public functions to operate on tensors (instead of ndarray)";
  let constructor = "mlir::npc::Numpy::createPublicFunctionsToTensorPass()";
}

#endif // NPC_NUMPY_PASSES