#ifndef NPC_E2E_PASSES
#define NPC_E2E_PASSES

include "mlir/Pass/PassBase.td"

def LowerLinalgOnTensorToLinalgOnMemref : 
    Pass<"lower-linalg-tensor-to-memref", "func::FuncOp"> {
  let summary = "Lowers linalg on tensors to linalg on memrefs";
  let constructor = "mlir::npc::createLowerLinalgOnTensorToLinalgOnMemrefPass()";
  let dependentDialects = [
    "linalg::LinalgDialect", "shape::ShapeDialect", "tcp::TCPDialect",
    "bufferization::BufferizationDialect"];
}

def LowerBroadcastToToLoops :
    Pass<"lower-broadcast-to-to-loops", "func::FuncOp"> {
  let summary = "Lower tcp::BroadcastTo to loops";
  let constructor = "mlir::npc::createLowerBroadcastToToLoopsPass()";
  let dependentDialects = ["tcp::TCPDialect"];
}

def LowerConstantTensorsToMemrefs :
    Pass<"lower-constant-tensors-to-memrefs", "ModuleOp">  {
  let summary = "Lower std.constant of tensor type to hybrid tensor/memref.";
  let description = [{
    This has to be a module pass since it involves creating tcp.global ops.
  }];
  let constructor = "mlir::npc::createLowerConstantTensorsToMemrefsPass()";
  let dependentDialects = ["tcp::TCPDialect", "bufferization::BufferizationDialect"];
}

def ResolveShapeOfOps : Pass<"resolve-shape-of-ops", "func::FuncOp"> {
  let summary = "Resolve shape.shape_of ops to other shapes.";
  let constructor = "mlir::npc::createResolveShapeOfOpsPass()";
  let dependentDialects = [
    "shape::ShapeDialect", "bufferization::BufferizationDialect", "cf::ControlFlowDialect"];
}

def ResolveTensorLoadStoreOps : Pass<"resolve-tensor-load-store-ops", "func::FuncOp"> {
  let summary = "Resolve tensor load/store ops";
  let constructor = "mlir::npc::createResolveTensorLoadStoreOpsPass()";
  let dependentDialects = ["memref::MemRefDialect", "linalg::LinalgDialect"];
}

def LowerLinalgLoopDimOps : Pass<"lower-linalg-loop-dim-ops", "func::FuncOp"> {
  let summary = "Lower dim ops introduced by linalg to loops lowering";
  let constructor = "mlir::npc::createLowerLinalgLoopDimOpsPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LowerRankedShapes : Pass<"lower-ranked-shapes", "func::FuncOp"> {
 let summary = "Lower ranked !shape.shape types to SSA values";
 let constructor = "mlir::npc::createLowerRankedShapesPass()";
 let dependentDialects = ["shape::ShapeDialect", "npcrt::NpcrtDialect"];
}

def LowerToNpcrtABI : Pass<"lower-to-npcrt-abi", "ModuleOp"> {
  let summary = "Lower tensors at ABI boundaries to npcrt dialect";
  let constructor = "mlir::npc::createLowerToNpcrtABIPass()";
  let dependentDialects = ["npcrt::NpcrtDialect", "linalg::LinalgDialect"];
}

def LowerAllocMemRefOps : Pass<"lower-alloc-memref-ops", "func::FuncOp"> {
  let summary = "Lower AllocMemrefOp's";
  let constructor = "mlir::npc::createLowerAllocMemRefOpsPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

//def LowerToLLVM : Pass<"e2e-lower-to-llvm", "ModuleOp"> {
//  let summary = "Lower everything to LLVM";
//  let constructor = "mlir::il::createLowerToLLVMPass()";
//}

#endif // NPC_E2E_PASSES
