#include "../PassDetail.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/ArrayRef.h"

#include "Dialect/TCP/IR/TCPOps.h"
#include <memory>

using namespace mlir;
using namespace mlir::npc;

namespace {
class ConvertAdd : public OpRewritePattern<tcp::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcp::AddOp op,
                                PatternRewriter &rewriter) const override {
    size_t rank = op.getType().cast<RankedTensorType>().getRank();
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    SmallVector<AffineMap, 3> accesses(/*args in + args out*/ 3,
                                       rewriter.getMultiDimIdentityMap(rank));

    Value resultInitTensor;
    auto input = op->getOperand(0);
    auto resultType = op->getResult(0).getType().cast<ShapedType>();
    if (!resultType.hasStaticShape()) {
      SmallVector<Value> dynamicDims;
      for (int i = 0; i < resultType.getRank(); i++) {
        dynamicDims.push_back(
            rewriter.create<tensor::DimOp>(op->getLoc(), input, i));
      }
      resultInitTensor = rewriter.create<tensor::EmptyOp>(
          op->getLoc(), resultType, dynamicDims);
    } else {
      resultInitTensor = rewriter.create<tensor::EmptyOp>(
          op->getLoc(), resultType.getShape(), resultType.getElementType());
    }

    auto genericOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(),
        /*result_tensors=*/TypeRange({op.getType()}),
        /*inputs=*/ValueRange({op.getLhs(), op.getRhs()}),
        /*outputs=*/resultInitTensor,
        /*indexing_maps=*/accesses,
        /*iterator_types=*/iterators);

    Region &region = genericOp.getRegion();
    Block *block = rewriter.createBlock(&region, region.begin());
    for (auto operandType : op->getOperandTypes()) {
      block->addArgument(operandType.cast<RankedTensorType>().getElementType(),
                         op.getLoc());
    }
    for (auto resultType : op->getResultTypes()) {
      block->addArgument(resultType.cast<RankedTensorType>().getElementType(),
                         op.getLoc());
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    Value bodyValue = rewriter.create<arith::AddFOp>(
        op->getLoc(), block->getArgument(0), block->getArgument(1));
    rewriter.create<linalg::YieldOp>(op->getLoc(), bodyValue);

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertTCPToLinalg : public ConvertTCPToLinalgBase<ConvertTCPToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    patterns.insert<ConvertAdd>(context);
    target.addIllegalOp<tcp::AddOp>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<BuiltinDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::npc::createConvertTCPToLinalgPass() {
  return std::make_unique<ConvertTCPToLinalg>();
}
