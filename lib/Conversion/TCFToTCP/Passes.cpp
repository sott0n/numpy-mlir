#include "../PassDetail.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/TCP/IR/TCPDialect.h"
#include "Dialect/TCF/IR/TCFOps.h"
#include "Dialect/TCP/IR/TCPOps.h"
#include "Conversion/TCFToTCP/Passes.h"

#include <algorithm>
#include <memory>

using namespace mlir;
using namespace mlir::npc;

namespace {
class ConvertAdd : public OpRewritePattern<tcf::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcf::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsType = op.getRhs().getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensors");

    Value lhsShape =
        rewriter.create<shape::ShapeOfOp>(op->getLoc(), op.getLhs());
    Value rhsShape =
        rewriter.create<shape::ShapeOfOp>(op->getLoc(), op.getRhs());
    Value resultType =
        rewriter.create<shape::ShapeOfOp>(op->getLoc(), op.getResult());

    Value broadcastedShape = rewriter.create<shape::BroadcastOp>(
        op.getLoc(), resultType.getType(), lhsShape, rhsShape,
        /*error=*/nullptr);
    rewriter.create<tcp::ShapeObserveErrorOp>(op->getLoc(), broadcastedShape);

    // TODO: It's annoying to do the dynamic broadcast above then
    // do the static transfer function here. Would be nice if they could
    // somehow be unified.
    SmallVector<int64_t, 6> broadcastedStaticShape;
    if (OpTrait::util::getBroadcastedShape(
            lhsType.getShape(), rhsType.getShape(), broadcastedStaticShape)) {
      auto elementType = RankedTensorType::get(broadcastedStaticShape,
                                               lhsType.getElementType());
      Value lhsBroadcasted = rewriter.create<tcp::BroadcastToOp>(
          op->getLoc(), elementType, op.getLhs(), broadcastedShape);
      Value rhsBroadcasted = rewriter.create<tcp::BroadcastToOp>(
          op->getLoc(), elementType, op.getRhs(), broadcastedShape);
      Value add = rewriter.create<tcp::AddOp>(op->getLoc(), op.getType(),
                                              lhsBroadcasted, rhsBroadcasted);
      rewriter.replaceOp(op, add);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class ConvertTCFToTCP : public ConvertTCFToTCPBase<ConvertTCFToTCP> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tcp::TCPDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertAdd>(context);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::npc::createConvertTCFToTCPPass() {
  return std::make_unique<ConvertTCFToTCP>();
}
