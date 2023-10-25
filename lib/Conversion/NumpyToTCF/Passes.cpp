#include "../PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/TCF/IR/TCFDialect.h"
#include "Dialect/Numpy/IR/NumpyOps.h"
#include "Dialect/TCF/IR/TCFOps.h"
#include "Conversion/NumpyToTCF/Passes.h"

#include <memory>

using namespace mlir;
using namespace mlir::npc;

namespace {
template <typename TargetTcfOp>
class ConvertBinaryBuiltinUfuncCallOp
    : public OpRewritePattern<Numpy::BuiltinUfuncCallOp> {
public:
  ConvertBinaryBuiltinUfuncCallOp(MLIRContext *context, StringRef qualifiedName,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), qualifiedName(qualifiedName) {}
  LogicalResult matchAndRewrite(Numpy::BuiltinUfuncCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getQualifiedName() != qualifiedName)
      return failure();
    if (op.getInputs().size() != 2)
      return failure();

    rewriter.replaceOpWithNewOp<TargetTcfOp>(
        op, op.getOutput().getType(), op.getInputs()[0], op.getInputs()[1]);
    return success();
  }

private:
  StringRef qualifiedName;
};
} // namespace

namespace {
class ConvertNumpyToTCF : public ConvertNumpyToTCFBase<ConvertNumpyToTCF> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tcf::TCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertBinaryBuiltinUfuncCallOp<tcf::AddOp>>(context,
                                                              "numpy.add");
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createConvertNumpyToTCFPass() {
  return std::make_unique<ConvertNumpyToTCF>();
}
