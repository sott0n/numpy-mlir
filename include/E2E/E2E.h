#ifndef NPC_E2E_E2E_H
#define NPC_E2E_E2E_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class ModuleOp;

namespace npc {

std::unique_ptr<OperationPass<func::FuncOp>>
createLowerLinalgOnTensorToLinalgOnMemrefPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createLowerBroadcastToToLoopsPass();

std::unique_ptr<OperationPass<ModuleOp>>
createLowerConstantTensorsToMemrefsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createResolveShapeOfOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createResolveTensorLoadStoreOpsPass();

//std::unique_ptr<OperationPass<func::FuncOp>> createLowerLinalgLoopDimOpsPass();
//
//std::unique_ptr<OperationPass<func::FuncOp>> createLowerRankedShapesPass();
//
//std::unique_ptr<OperationPass<ModuleOp>> createLowerToIlrtABIPass();
//
//std::unique_ptr<OperationPass<func::FuncOp>> createLowerAllocMemRefOpsPass();
//
//std::unique_ptr<OperationPass<ModuleOp>> createLowerToLLVMPass();
//
void createLowerToHybridTensorMemRefPipeline(OpPassManager &pm);

struct E2ELoweringPipelineOptions
    : public PassPipelineOptions<E2ELoweringPipelineOptions> {
  // If this options is true, then perform optimizations.
  // If this options is false, only do the bare minimum for correctness.
  Option<bool> optimize{*this, "optimize", llvm::cl::desc("Do optimizations."),
                        llvm::cl::init(false)};
};

// The main pipeline tha encapsulates the full E2E lowering.
void createE2ELoweringPipeline(OpPassManager &pm,
                               const E2ELoweringPipelineOptions &options);

} // namespace npc
} // namespace mlir

#endif // NPC_E2E_E2E_H
