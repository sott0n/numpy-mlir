/// This is the base file for our "end-to-end" il lowering pipeline.
/// At the moment, the first `end` is TCF ops and the second `end` is `llvm`
/// dialect suitable for jitting.

#include "E2E/E2E.h"
#include "PassDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "Dialect/TCP/IR/TCPOps.h"
#include "Conversion/TCFToTCP/Passes.h"
#include "Conversion/TCPToLinalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::npc;

//===----------------------------------------------------------------------===//
// ResolveShapeOfOps
//===----------------------------------------------------------------------===//

namespace {
class ResolveShapeOfOpViaAllocMemRefOp
    : public OpRewritePattern<shape::ShapeOfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter &rewriter) const override {
    if (auto toTensor = llvm::dyn_cast_or_null<bufferization::ToTensorOp>(
            op.getOperand().getDefiningOp())) {
      if (auto allocMemRef = llvm::dyn_cast_or_null<tcp::AllocMemRefOp>(
              toTensor.getOperand().getDefiningOp())) {
        rewriter.replaceOp(op, allocMemRef.getOperand(0));
        return success();
      }
    }
    return failure();
  }
};
} // namespace

namespace {
class ResolveShapeOfOps : public ResolveShapeOfOpsBase<ResolveShapeOfOps> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<ResolveShapeOfOpViaAllocMemRefOp>(context);
    ConversionTarget target(*context);
    target.addLegalOp<shape::ConstShapeOp>();
    target.addDynamicallyLegalOp<shape::ShapeOfOp>(
        [](shape::ShapeOfOp shapeOfOp) {
          if (auto blockArg =
                  shapeOfOp.getOperand().dyn_cast<BlockArgument>()) {
            Block *block = blockArg.getOwner();
            if (&block->getParent()->front() == block) {
              return true;
            }
          }
          return false;
        });

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createResolveShapeOfOpsPass() {
  return std::make_unique<ResolveShapeOfOps>();
}

//===----------------------------------------------------------------------===//
// ResolveTensorLoadStoreOps
//===----------------------------------------------------------------------===//

namespace {
class ReplaceTensorStoreWithCopyPattern
    : public OpRewritePattern<memref::TensorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::TensorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorLoad = op.getTensor().getDefiningOp<bufferization::ToTensorOp>();
    if (!tensorLoad)
      return rewriter.notifyMatchFailure(op, "not fed by tensor_load op");
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, tensorLoad.getMemref(),
                                                op.getMemref());

    return success();
  }
};
} // namespace

namespace {
class EraseUnusedTensorLoadOpPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.use_empty())
      return rewriter.notifyMatchFailure(op, "has uses");
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class ResolveTensorLoadStoreOps
    : public ResolveTensorLoadStoreOpsBase<ResolveTensorLoadStoreOps> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<ReplaceTensorStoreWithCopyPattern>(context);
    patterns.insert<EraseUnusedTensorLoadOpPattern>(context);
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalOp<bufferization::ToTensorOp>();
    target.addDynamicallyLegalOp<memref::TensorStoreOp>(
        [](memref::TensorStoreOp op) {
          return op.getTensor().isa<BlockArgument>();
        });
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createResolveTensorLoadStoreOpsPass() {
  return std::make_unique<ResolveTensorLoadStoreOps>();
}

//===----------------------------------------------------------------------===//
// LowerLinalgLoopDimOps
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgLoopDimOp : public OpRewritePattern<memref::DimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DimOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Remove this const pattern when lowering to shape.get_extent.
    auto constIndex = op.getConstantIndex();
    if (!constIndex)
      return failure();

    auto allocMemRef = op.getSource().getDefiningOp<tcp::AllocMemRefOp>();
    if (!allocMemRef)
      return rewriter.notifyMatchFailure(op, "could not find alloc_memref");
    rewriter.replaceOpWithNewOp<tcp::GetExtentOp>(
        op, allocMemRef.getShape().front(), *constIndex);
    return success();
  }
};

} // namespace

namespace {
class LowerLinalgLoopDimOps
    : public LowerLinalgLoopDimOpsBase<LowerLinalgLoopDimOps> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<LowerLinalgLoopDimOp>(context);
    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<memref::DimOp>([](memref::DimOp op) -> bool {
      // TODO: We only need this because we use `dim` ops for the memref ABI.
      // Onse we layer that out into our own runtime types, we can remove this.
      return !op.getSource().getDefiningOp<tcp::AllocMemRefOp>();
    });
    target.addLegalOp<tcp::GetExtentOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createLowerLinalgLoopDimOpsPass() {
  return std::make_unique<LowerLinalgLoopDimOps>();
}

//===----------------------------------------------------------------------===//
// LowerAllocMemRefOps
//===----------------------------------------------------------------------===//

namespace {
class LowerAllocMemRefOp : public OpRewritePattern<tcp::AllocMemRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcp::AllocMemRefOp op,
                                PatternRewriter &rewriter) const override {
    auto memrefType = op.getType().cast<MemRefType>();
    auto shape = op.getOperand(0);
    // std::alloc only accepts the dynamic extents as operands, so only
    // collect those.
    SmallVector<Value, 6> dynamicExtents;
    for (int64_t i = 0, e = memrefType.getRank(); i < e; i++) {
      if (memrefType.isDynamicDim(i)) {
        auto extent = rewriter.create<tcp::GetExtentOp>(op.getLoc(), shape, i);
        dynamicExtents.push_back(extent);
      }
    }
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType,
                                                 dynamicExtents);
    return success();
  }
};
} // namespace

namespace {
class LowerAllocMemRefOps
    : public LowerAllocMemRefOpsBase<LowerAllocMemRefOps> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<LowerAllocMemRefOp>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<tcp::AllocMemRefOp>();
    target.addLegalOp<tcp::GetExtentOp>();
    target.addLegalOp<memref::AllocOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createLowerAllocMemRefOpsPass() {
  return std::make_unique<LowerAllocMemRefOps>();
}

//===----------------------------------------------------------------------===//
// createE2ELoweringPipeline
//===----------------------------------------------------------------------===//

void mlir::npc::createE2ELoweringPipeline(
    OpPassManager &pm, const E2ELoweringPipelineOptions &options) {
  // Input IR is TCF ops.

  // Convert to TCP.
  pm.addPass(createConvertTCFToTCPPass());

  // --------------------------------------------------------------------------
  // Tensor to buffer (memref) conversion.
  // --------------------------------------------------------------------------

  // Convert tcp ops to Linalg where possible, as we want generic linalg
  // tensor->memref to do most of the mechanical work of rewriting ops in
  // terms of tensors to ops in terms of memrefs (since it is easy on that
  // representation).
  pm.addPass(createConvertTCPToLinalgPass());

  // Lower to hybrid tensor/memref
  //
  // The hybrid tensor/memref representation gurantees:
  // - every use of a tensor is a tensor_store op writing it into a memref
  // - every def of a tensor is a tensor_load op loading out of some memref.
  // - every memref is allocated by a `tcp.alloc_memref(%shape)` op.
  // - every memref is only ever written once, and never mutated.
  //
  // Exceptions: "boundaries" such as function arguments and island
  // live-outs.
  //
  // Or, another way to say this: the hybrid tensor/memref representation
  // doesn't attempt to eliminate the original tensors from the program,
  // but rather locally expands operations on tensors to be small subgraphs
  // with tensor_load/tensor_store at the boundaries, leaving enough
  // invariants that we can clean it up later.
  //
  // The core invariants that are needed for this step are that the
  // tensor-level ops we receive as input have a way of calculating the
  // sizes for their outputs. This is equivalent to saving that
  // `shape.shape_of` on the result of an op must be calculatable in terms
  // of the shapes of the inputs to the op.
  createLowerToHybridTensorMemRefPipeline(pm);

  // At this point, the invariants of the hybrid tensor/memref
  // representation allow us to resolve `shape.shape_of` ops to shape
  // computations earlier in the program. Specifically, every
  // `shape.shape_of` can be resolved to the shape argument to the
  // corresponding `tcp.alloc_memref` op of the tensor_load that produced
  // that tensor.
  pm.addNestedPass<func::FuncOp>(createResolveShapeOfOpsPass());

  // Now, we use the hybrid tensor/memref invariants to replace the
  // tensor_store ops with memref copy operations and erase the
  // tensor_load/tensor_store ops.
  pm.addNestedPass<func::FuncOp>(createResolveTensorLoadStoreOpsPass());

  //// We need to finalize the removal of tensors from the program. To do
  //// that, we need to interface with a runtime ABI.
  //// We have a specialized dialect npc-rt which models the runtime data
  //// stuructures, and function signatures (and presumably eventually, other
  //// ABI boundaries like external calls if we ever support it) will be
  //// converted.
  pm.addPass(createLowerToNpcrtABIPass());

  // At this point, we have loose shape calculations floating around, so
  // it's a good time to do some general cleanups.
  if (options.optimize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Preparation for converting to LLVM module.
  // --------------------------------------------------------------------------

  // Lower Linalg ops to loops.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

  // Lowering linalg to loops introduces `dim` ops. Here we look through
  // use-def chains to find `tcp.alloc_memref` ops that we can get a shape
  // out of.
  // Currently, this is trivial, but after more aggressive buffer
  // allocation optimizations or linalg tiling this step will need to look
  // through slices/views and stuff.
  pm.addNestedPass<func::FuncOp>(createLowerLinalgLoopDimOpsPass());

  // AllocMemRefOp's take a `!shape.shape` as an argument. We need to
  // resolve this to individual extents before we lower ranked shapes.
  pm.addNestedPass<func::FuncOp>(createLowerAllocMemRefOpsPass());

  //// Lower shapes to SSA values.
  //// This replaces all tcf::GetExtentOp's with explicit SSA computations
  //// for the scalar extent. This requires shapes which are ranked. Any
  //// unranked shapes will need to be handled by a runtime shape type,
  //// through we don't currently support that.
  ////
  //// At this point, in the case of programs with only ranked shapes, all
  //// !shape.shape types will be gone.
  //pm.addNestedPass<func::FuncOp>(createLowerRankedShapesPass());

  //// Run a some cleanups.
  //if (options.optimize) {
  //  pm.addPass(createCanonicalizerPass());
  //  pm.addPass(createCSEPass());
  //}

  //// --------------------------------------------------------------------------
  //// Final conversion to an LLVM modules.
  //// --------------------------------------------------------------------------

  //// Convert scf to std control flow in preparation for going to LLVM.
  //pm.addPass(createConvertSCFToCFPass());

  //// Finally, convert to LLVM dialect using our custom LowerToLLVM pass
  //// which reuses the upstream patterns and gives us a place to add our own
  //// patterns for any custom ops and types we wish to lower.
  //pm.addPass(createLowerToLLVMPass());

  //// Although LLVM will clean everything up eventually, for the sake of IR
  //// clarity while still in MLIR, run some cleanups.
  //if (options.optimize) {
  //  pm.addPass(createCanonicalizerPass());
  //  pm.addPass(createCSEPass());
  //}
}
