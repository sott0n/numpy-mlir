#include "PassDetail.h"
#include "E2E/E2E.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/Npcrt/IR/NpcrtOps.h"
#include "Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::npc;

namespace {
class LowerConstShapeOp : public OpConversionPattern<shape::ConstShapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::ConstShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto extents = llvm::to_vector<6>(llvm::map_range(
        op.getShape().getValues<int64_t>(), [&](int64_t extent) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), extent);
        }));
    rewriter.replaceOpWithNewOp<shape::FromExtentsOp>(
        op, rewriter.getType<shape::ShapeType>(), extents);
    return success();
  }
};
} // namespace

namespace {
class LowerShapeBroadcastOp : public OpConversionPattern<shape::BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getOperands()[0].getDefiningOp<shape::FromExtentsOp>();
    auto rhs = adaptor.getOperands()[1].getDefiningOp<shape::FromExtentsOp>();
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(op, "operands not converted");
    // Establish invariant that rank(lhs) >= rank(rhs).
    if (lhs.getExtents().size() < rhs.getExtents().size())
      std::swap(lhs, rhs);
    auto rankDiscrepancy = lhs.getExtents().size() - rhs.getExtents().size();

    // Helper that creates IR
    // ```
    // abort_if(extent != resultExtent && extent != 1)
    // ```
    // This is the numpy broadcasting legality check.
    auto createAbortIfIllegalBroadcastExtent = [&](Value extent,
                                                   Value resultExtent) {
      auto c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
      auto extentNeMax = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, extent, resultExtent);
      auto extentNeOne = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, extent, c1);
      auto bothTrue =
          rewriter.create<arith::AndIOp>(op.getLoc(), extentNeMax, extentNeOne);
      rewriter.create<npcrt::AbortIfOp>(op.getLoc(), bothTrue);
    };

    auto resultExtents = llvm::to_vector<6>(lhs.getExtents());
    for (int i = 0, e = rhs.getExtents().size(); i < e; i++) {
      auto lhsExtent = lhs.getExtents()[rankDiscrepancy + i];
      auto rhsExtent = rhs.getExtents()[i];
      auto ugt = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ugt, lhsExtent, rhsExtent);
      auto max = rewriter.create<arith::SelectOp>(op.getLoc(), ugt, lhsExtent,
                                                  rhsExtent);
      auto &resultExtent = resultExtents[rankDiscrepancy + i];
      resultExtent = max;
      createAbortIfIllegalBroadcastExtent(lhsExtent, resultExtent);
      createAbortIfIllegalBroadcastExtent(rhsExtent, resultExtent);
    }
    rewriter.replaceOpWithNewOp<shape::FromExtentsOp>(
        op, shape::ShapeType::get(rewriter.getContext()), resultExtents);
    return success();
  }
};
} // namespace

namespace {
// Rewrite `get_extent(from_extents(x1,x2,x3), N) -> xN`
class LowerShapeGetExtentOp : public OpConversionPattern<tcp::GetExtentOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GetExtentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fromExtents = adaptor.getShape().getDefiningOp<shape::FromExtentsOp>();
    if (!fromExtents)
      return rewriter.notifyMatchFailure(op, "not a from_extents op");
    int64_t dim = op.getDim();
    rewriter.replaceOp(op, ValueRange(fromExtents.getExtents())[dim]);
    return success();
  }
};
} // namespace

namespace {
// Now that we have lowered ranked shapes, which reifies the eager
// error-handling code, the tcp::ShapeObserveErrorOp's are no longer
// needed.
class EraseShapeObserveErrorOp
    : public OpConversionPattern<tcp::ShapeObserveErrorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::ShapeObserveErrorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class LowerRankedShapes : public LowerRankedShapesBase<LowerRankedShapes> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<LowerConstShapeOp>(context);
    patterns.insert<LowerShapeBroadcastOp>(context);
    patterns.insert<LowerShapeGetExtentOp>(context);
    patterns.insert<EraseShapeObserveErrorOp>(context);

    ConversionTarget target(*context);
    target.addIllegalOp<shape::ShapeOfOp>();
    target.addIllegalOp<shape::BroadcastOp>();
    target.addIllegalOp<tcp::GetExtentOp>();
    target.addIllegalOp<tcp::ShapeObserveErrorOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<shape::FromExtentsOp>();
    target.addLegalOp<npcrt::AbortIfOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Erase some stray shape ops from the program. They can't be
    // deleted during convension because they become unused only after
    // subsequent patterns bypass them.
    auto walkResult = func.walk([](Operation *op) {
      if (!isa<shape::FromExtentsOp>(op))
        return WalkResult::advance();
      if (!op->use_empty()) {
        op->emitError("could not be eliminated");
        return WalkResult::advance();
      }
      op->erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createLowerRankedShapesPass() {
  return std::make_unique<LowerRankedShapes>();
}
