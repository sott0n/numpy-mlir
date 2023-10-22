#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include "Dialect/Numpy/IR/NumpyDialect.h"
#include "Dialect/Numpy/IR/NumpyOps.h"
#include "Dialect/Numpy/Transforms/Passes.h"

#include <memory>

using namespace mlir;
using namespace mlir::npc::Numpy;

namespace {

class PublicFunctionsToTensorPass
    : public NumpyPublicFunctionsToTensorBase<PublicFunctionsToTensorPass> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](func::FuncOp func) {
      if (func.getVisibility() != SymbolTable::Visibility::Public)
        return;
      if (func.isExternal())
        return;
      auto uses = SymbolTable::getSymbolUses(func, module);
      if (!uses || uses->begin() != uses->end()) {
        func->emitWarning() << "unimplemented: cannot convert ndarray->tensor "
                            << "signature for public function with uses";
        return;
      }
      rewriteSignature(func);
    });
  }

  void rewriteSignature(func::FuncOp func) {
    auto &entryBlock = func.getBody().front();
    auto funcType = func.getFunctionType();
    auto loc = func.getLoc();

    // Rewrite inputs.
    auto builder = OpBuilder::atBlockBegin(&entryBlock);
    auto inputTypes = llvm::to_vector<4>(funcType.getInputs());
    for (unsigned i = 0; i < inputTypes.size(); ++i) {
      auto arrayType = inputTypes[i].dyn_cast<NdArrayType>();
      if (!arrayType)
        continue;
      Type tensorType = arrayType.toTensorType();
      BlockArgument argument = entryBlock.getArgument(i);
      argument.setType(tensorType);
      auto createOp =
          builder.create<CreateArrayFromTensorOp>(loc, arrayType, argument);
      argument.replaceAllUsesExcept(createOp,
                                    SmallPtrSet<Operation *, 1>{createOp});
      inputTypes[i] = tensorType;
    }

    // Rewrite result signature.
    auto resultTypes = llvm::to_vector<4>(funcType.getResults());
    for (auto &resultType : resultTypes) {
      auto arrayType = resultType.dyn_cast<NdArrayType>();
      if (arrayType)
        resultType = arrayType.toTensorType();
    }

    // Update signature.
    funcType =
        FunctionType::get(funcType.getContext(), inputTypes, resultTypes);
    func.setType(funcType);

    // Rewrite all return terminators.
    func.walk([&](func::ReturnOp term) {
      OpBuilder builder(term);
      for (unsigned i = 0; i < term->getNumOperands(); ++i) {
        Value operand = term->getOperand(i);
        auto arrayType = operand.getType().dyn_cast<NdArrayType>();
        if (!arrayType)
          continue;
        Type tensorType = arrayType.toTensorType();
        auto copyOp = builder.create<CopyToTensorOp>(loc, tensorType, operand);
        term->setOperand(i, copyOp);
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::npc::Numpy::createPublicFunctionsToTensorPass() {
  return std::make_unique<PublicFunctionsToTensorPass>();
}
