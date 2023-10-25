#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/TCF/IR/TCFDialect.h"
#include "Dialect/TCF/IR/TCFOps.h"
#include "Dialect/TCF/Transforms/Passes.h"

#include <memory>

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::tcf;

namespace {

class ShapeRefinementPass : public TCFShapeRefinementBase<ShapeRefinementPass> {
  void runOnOperation() override {
    auto func = getOperation();
    func.walk([](tcf::AddOp addOp) {
      auto lhsType = addOp.getLhs().getType();
      auto rhsType = addOp.getRhs().getType();
      if (lhsType == rhsType) {
        addOp.getResult().setType(lhsType);
      }
    });

    // If the change cascaded to any returns, need to update the function
    // signature.
    std::optional<func::ReturnOp> firstReturnOp;
    func.walk([&](func::ReturnOp returnOp) {
      if (!firstReturnOp) {
        firstReturnOp = returnOp;
      } else {
        if (returnOp->getOperandTypes() != firstReturnOp->getOperandTypes()) {
          returnOp.emitError() << "after refining shapes, different "
                                  "terminators have different types";
          signalPassFailure();
        }
      }
    });

    assert(firstReturnOp && "function lacks a terminator");
    auto funcType = func.getFunctionType();
    SmallVector<Type, 4> resultTypes(firstReturnOp->getOperandTypes().begin(),
                                     firstReturnOp->getOperandTypes().end());
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   resultTypes));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::tcf::createShapeRefinementPass() {
  return std::make_unique<ShapeRefinementPass>();
}
