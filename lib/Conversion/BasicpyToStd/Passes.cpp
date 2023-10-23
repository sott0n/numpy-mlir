#include "../PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Conversion/BasicpyToStd/Passes.h"
#include "Conversion/BasicpyToStd/Patterns.h"
#include <memory>

using namespace mlir;
using namespace mlir::npc;

namespace {

class ConvertBasicpyToStd
    : public ConvertBasicpyToStdBase<ConvertBasicpyToStd> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    populateBasicpyToStdPrimitiveOpPatterns(context, patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::createConvertBasicpyToStdPass() {
  return std::make_unique<ConvertBasicpyToStd>();
}
