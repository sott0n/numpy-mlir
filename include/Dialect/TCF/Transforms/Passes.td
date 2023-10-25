#ifndef NPC_TCF_PASSES
#define NPC_TCF_PASSES

include "mlir/Pass/PassBase.td"

def TCFShapeRefinement : Pass<"tcf-shape-refinement", "func::FuncOp"> {
  let summary = "Refines shapes of tensors";
  let constructor = "mlir::npc::tcf::createShapeRefinementPass()";
}

#endif // NPC_TCF_PASSES