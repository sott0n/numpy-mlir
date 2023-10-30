#include "PassDetail.h"
#include "E2E/E2E.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/TCP/IR/TCPDialect.h"
#include "Dialect/TCP/IR/TCPOps.h"
#include "Conversion/TCFToTCP/Passes.h"
#include "Conversion/TCPToLinalg/Passes.h"

using namespace mlir;
using namespace mlir::npc;

static Value allocMemRefForTensor(OpBuilder &builder, Value tensor, Value shape,
                                  Location loc) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return builder.create<tcp::AllocMemRefOp>(loc, memrefType, shape);
}

//===----------------------------------------------------------------------===//
// LowerBroadcastTo
//===----------------------------------------------------------------------===//

namespace {
class LowerBroadcastToToLoopsPattern
    : public OpRewritePattern<tcp::BroadcastToOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcp::BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<RankedTensorType>();
    auto inputType = op.getOperand().getType().cast<RankedTensorType>();
    Value resultMemref = rewriter.create<tcp::AllocMemRefOp>(
        op.getLoc(),
        MemRefType::get(resultType.getShape(), resultType.getElementType()),
        op.getShape());
    Value inputMemref = allocMemRefForTensor(
        rewriter, op.getOperand(),
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.getOperand()),
        op.getLoc());
    rewriter.create<memref::TensorStoreOp>(op.getLoc(), op.getOperand(),
                                           inputMemref);
    SmallVector<Value, 6> outputExtents;
    SmallVector<Value, 6> inputDimRequiresBroadcasting;

    for (int i = 0, e = resultType.getRank(); i < e; i++) {
      Value outputExtent = rewriter.create<tcp::GetExtentOp>(
          op.getLoc(), op.getShape(), rewriter.getI64IntegerAttr(i));
      outputExtents.push_back(outputExtent);
    }
    int rankDiff = resultType.getRank() - inputType.getRank();
    for (int i = 0, e = inputType.getRank(); i < e; i++) {
      // Calculate the relevant extents.
      Value inputExtent =
          rewriter.create<memref::DimOp>(op.getLoc(), op.getOperand(), i);
      inputDimRequiresBroadcasting.push_back(rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, inputExtent,
          outputExtents[rankDiff + i]));
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);

      SmallVector<Value, 6> inductionVariables;
      // Create the (perfectly nested) loops.
      // Loop invariant: At the start of iteration `i`, the rewriter insertion
      // point is inside `i` nested loops.
      for (int i = 0, e = resultType.getRank(); i < e; i++) {
        auto loop = rewriter.create<scf::ForOp>(
            op.getLoc(), c0, outputExtents[i], c1, ValueRange({}));
        Block *body = loop.getBody();
        inductionVariables.push_back(body->getArgument(0));
        // Leave the insertion point at the beginning of the body.
        rewriter.setInsertionPointToStart(body);
      }

      // Create the inner loop body.
      // When reading from the input, clamp any indices for dimensions that are
      // being broadcast.
      SmallVector<Value, 6> inputIndices;
      for (int i = 0, e = inputType.getRank(); i < e; i++) {
        auto c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto select = rewriter.create<arith::SelectOp>(
            op.getLoc(), inputDimRequiresBroadcasting[i], c0,
            inductionVariables[rankDiff + i]);
        inputIndices.push_back(select);
      }
      Value load = rewriter.create<memref::LoadOp>(op.getLoc(), inputMemref,
                                                   inputIndices);
      rewriter.create<memref::StoreOp>(op.getLoc(), load, resultMemref,
                                       inductionVariables);
    }

    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resultMemref);
    return success();
  }
};
} // namespace

// We currently only create DimOp's during LoweringBroadcastToToLoopsPattern,
// so for now just stuff it in here.
namespace {
class LowerDimOpToShape : public OpRewritePattern<memref::DimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DimOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Remove this const pattern when lowering to shape.get_extent.
    auto constIndex = op.getConstantIndex();
    if (!constIndex)
      return failure();
    auto shape = rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.getSource());
    rewriter.replaceOpWithNewOp<tcp::GetExtentOp>(op, shape, *constIndex);
    return success();
  }
};
} // namespace

namespace {
class LowerBroadcastToToLoops
    : public LowerBroadcastToToLoopsBase<LowerBroadcastToToLoops> {
  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tcp::TCPDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();

    RewritePatternSet patterns(context);
    target.addIllegalOp<tcp::BroadcastToOp>();
    patterns.insert<LowerBroadcastToToLoopsPattern>(context);
    target.addIllegalOp<memref::DimOp>();
    patterns.insert<LowerDimOpToShape>(context);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createLowerBroadcastToToLoopsPass() {
  return std::make_unique<LowerBroadcastToToLoops>();
}

//===----------------------------------------------------------------------===//
// LowerLinalgOnTensorToLinalgMemref
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgGenericTensorToMemref
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle generic ops where all operands and results are tensors.
    if (!llvm::all_of(op.getOperandTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); })) {
      return rewriter.notifyMatchFailure(op, "all operands must be tensors");
    }
    if (!llvm::all_of(op.getResultTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); })) {
      return rewriter.notifyMatchFailure(op, "all results must be tensors");
    }

    if (!llvm::all_of(op.getIndexingMaps(), [](Attribute map) {
          return map.cast<AffineMapAttr>().getValue().isIdentity();
        })) {
      return rewriter.notifyMatchFailure(
          op, "all indexing maps must be identity maps");
    }
    if (!llvm::all_of(op.getIteratorTypes(), [](Attribute it) {
          return linalg::isParallelIterator(
              it.cast<linalg::IteratorTypeAttr>().getValue());
        })) {
      return rewriter.notifyMatchFailure(
          op, "all iterator types must be `parallel`");
    }

    SmallVector<Value, 6> memrefs;
    SmallVector<Value, 6> resultMemrefs;
    SmallVector<Value, 6> operandShapes;
    for (auto tensor : op.getInputs()) {
      auto shape = rewriter.create<shape::ShapeOfOp>(op.getLoc(), tensor);
      auto memref = allocMemRefForTensor(rewriter, tensor, shape, op.getLoc());
      rewriter.create<memref::TensorStoreOp>(op.getLoc(), tensor, memref);
      memrefs.push_back(memref);
      operandShapes.push_back(shape);
    }
    auto shapeType = shape::ShapeType::get(rewriter.getContext());
    SmallVector<Type, 6> shapeTypes(op.getNumResults(), shapeType);
    SmallVector<Value, 6> resultShapes(op.getNumResults(), operandShapes[0]);
    for (auto t : llvm::zip(op.getResults(), resultShapes)) {
      auto tensor = std::get<0>(t);
      auto shape = std::get<1>(t);
      auto memref = allocMemRefForTensor(rewriter, tensor, shape, op.getLoc());
      memrefs.push_back(memref);
      resultMemrefs.push_back(memref);
    }

    auto newGeneric = rewriter.create<linalg::GenericOp>(
        op.getLoc(), std::nullopt, ValueRange(memrefs), op->getAttrs());
    newGeneric.getRegion().getBlocks().clear();
    IRMapping mapper;
    op.getRegion().cloneInto(&newGeneric.getRegion(), mapper);

    // Erase InitTensor for outputs.
    auto prevOutputs = op.getOutputs();
    if (auto emptyOp = llvm::dyn_cast_or_null<tensor::EmptyOp>(
            prevOutputs[0].getDefiningOp())) {
      for (int i = 0; (size_t)i < emptyOp.getOperands().size(); i++) {
        if (auto dimOp = llvm::dyn_cast_or_null<tensor::DimOp>(
                emptyOp.getOperand(i).getDefiningOp())) {
          if (auto constantOp = llvm::dyn_cast_or_null<arith::ConstantOp>(
                  dimOp.getOperand(1).getDefiningOp())) {
            rewriter.eraseOp(constantOp);
          }
          rewriter.eraseOp(dimOp);
        }
      }
      rewriter.eraseOp(emptyOp);
    }

    auto newResultTensors =
        llvm::to_vector<6>(llvm::map_range(resultMemrefs, [&](Value memref) {
          return rewriter.create<bufferization::ToTensorOp>(op.getLoc(), memref)
              .getResult();
        }));
    llvm::ArrayRef<Value> refNewResultTensor(newResultTensors.data(),
                                             newResultTensors.size());
    ValueRange newResultTensorRange(refNewResultTensor);
    rewriter.replaceOp(op, newResultTensorRange);

    return success();
  }
};
} // namespace

namespace {
class LowerLinalgOnTensorToLinalgOnMemref
    : public LowerLinalgOnTensorToLinalgOnMemrefBase<
          LowerLinalgOnTensorToLinalgOnMemref> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalOp<tcp::AllocMemRefOp>();
    patterns.insert<LowerLinalgGenericTensorToMemref>(context);
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      if (llvm::any_of(op.getOperandTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        return false;
      }
      if (llvm::any_of(op.getResultTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        return false;
      }
      return true;
    });

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createLowerLinalgOnTensorToLinalgOnMemrefPass() {
  return std::make_unique<LowerLinalgOnTensorToLinalgOnMemref>();
}

//===----------------------------------------------------------------------===//
// LowerConstantTensorToMemrefs
//===----------------------------------------------------------------------===//

namespace {
// This class creates global ops for all tensor-valued constants in the program.
// It creates them with pretty names and makes sure that duplicate globals
// aren't created.
class GlobalCreator {
public:
  explicit GlobalCreator(ModuleOp module);
  tcp::GlobalOp getGlobalFor(Attribute attr) {
    assert(globals.find(attr) != globals.end() && "unknown constant attr");
    return globals[attr];
  }

private:
  DenseMap<Attribute, tcp::GlobalOp> globals;
};

GlobalCreator::GlobalCreator(ModuleOp module) {
  // Create a builder without an insertion point. We will insert using the
  // symbol table to gurantee unique names.
  OpBuilder globalBuilder(module.getContext());
  SymbolTable symbolTable(module);
  module.walk([&](arith::ConstantOp op) {
    // We only want tensor constants for now.
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return;
    // If we already have a global for this constant value, no need to do
    // anything else.
    auto it = globals.find(op.getValue());
    if (it != globals.end())
      return;

    // Create a pretty name.
    SmallString<64> buf;
    llvm::raw_svector_ostream os(buf);
    interleave(type.getShape(), os, "x");
    os << "x" << type.getElementType();

    auto global = globalBuilder.create<tcp::GlobalOp>(
        op.getLoc(), (Twine("__constant_") + os.str()).str(),
        op.getValue().cast<ElementsAttr>());
    symbolTable.insert(global);
    // The symbol table insert at the end of the module, but globals are a bit
    // nicer if they are at the beginning.
    global.getOperation()->moveBefore(&module.front());
    globals[op.getValue()] = global;
  });
}
} // namespace

namespace {
class LowerConstantTensorsToMemrefs
    : public LowerConstantTensorsToMemrefsBase<LowerConstantTensorsToMemrefs> {
  void runOnOperation() override {
    auto module = getOperation();
    GlobalCreator globals(module);

    // With the global traversal factored into GlobalCreator, this could in
    // principle be done with a pattern.
    module.walk([&](arith::ConstantOp op) {
      auto type = op.getType().dyn_cast<RankedTensorType>();
      if (!type)
        return;
      auto global = globals.getGlobalFor(op.getValue());
      OpBuilder builder(op);
      auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
      auto memref = builder.create<tcp::GetGlobalMemrefOp>(
          op.getLoc(), memrefType, global.getName());
      Value tensor =
          builder.create<bufferization::ToTensorOp>(op.getLoc(), type, memref);
      op.replaceAllUsesWith(tensor);
      op.erase();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::npc::createLowerConstantTensorsToMemrefsPass() {
  return std::make_unique<LowerConstantTensorsToMemrefs>();
}

void mlir::npc::createLowerToHybridTensorMemRefPipeline(OpPassManager &pm) {
  // Lower to hybrid tensor/memref.
  // The invariant of "hybrid tensor/memref" is that the core computation
  // ops operate on memref, but we launder in and out of tensors in such a
  // way that the original SSA tensor values remain and can be traced to
  // their corresponding memrefs (via tensor_load/tensor_store) which are
  // allocated with alloc_shape ops.
  // Thus, shape.shape_of ops on the original tensors in the program can be
  // resolved to the shapes in the alloc_memref calls.
  pm.addPass(createLowerConstantTensorsToMemrefsPass());

  pm.addNestedPass<func::FuncOp>(
      createLowerLinalgOnTensorToLinalgOnMemrefPass());
  pm.addNestedPass<func::FuncOp>(createLowerBroadcastToToLoopsPass());
}
