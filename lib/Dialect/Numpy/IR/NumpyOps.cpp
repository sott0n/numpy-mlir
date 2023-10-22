#include "Dialect/Numpy/IR/NumpyOps.h"
#include "Dialect/Numpy/IR/NumpyDialect.h"
#include "Dialect/Basicpy/IR/BasicpyDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::Numpy;

//===----------------------------------------------------------------------===//
// Type inference
//===----------------------------------------------------------------------===//

/// Adds constraints to relating a unary op that accepts and returns either
/// tensor or ndarray types where the dtype should be the same.
/// Type constraints are added on the dtype, not the outer object type.
//static void constraintUnaryDtypeInvariantOp(Typing::CPA::Context &context,
//                                            Value source, Value dest,
//                                            Operation *op) {
//  auto &env = context.getCurrentEnvironment();
//  auto *sourceTn =
//      llvm::dyn_cast<Typing::CPA::ObjectValueType>(env.mapValueToType(source));
//  auto *destTn =
//      llvm::dyn_cast<Typing::CPA::ObjectValueType>(env.mapValueToType(dest));
//  if (sourceTn && destTn && sourceTn->getFieldCount() == 1 &&
//      destTn->getFieldCount() == 1) {
//    context.getConstraint(sourceTn->getFieldTypes().front(),
//                          destTn->getFieldTypes().front());
//  }
//}
//
//void CreateArrayFromTensorOp::addCPAConstraints(Typing::CPA::Context &context) {
//  constraintUnaryDtypeInvariantOp(context, getSource(), getDest(), *this);
//}
//
//void CopyToTensorOp::addCPAConstraints(Typing::CPA::Context &context) {
//  constraintUnaryDtypeInvariantOp(context, getSource(), getDest(), *this);
//}
//
//void BuiltinUfuncCallOp::addCPAConstraints(Typing::CPA::Context &context) {
//  // TODO: This should really be a function call chosen so as to promote
//  // arguments. For now, though, we just say that the result is constrained
//  // to the inputs. Note that not all ufuncs transfer types like this.
//  // We just pretend this is two unary functions that write into the output.
//  for (auto input : getInputs()) {
//    constraintUnaryDtypeInvariantOp(context, input, getOutput(), *this);
//  }
//}
//
////===----------------------------------------------------------------------===//
//// CreateArrayFromTensorOp
////===----------------------------------------------------------------------===//
//
//namespace {
///// Match create_array_from_tensor -> copy_to_tensor and elide in favor
///// of the original tensor.
//class ElideCreateRedundantArrayFromTensor
//    : public OpRewritePattern<CopyToTensorOp> {
//public:
//  using OpRewritePattern::OpRewritePattern;
//  LogicalResult matchAndRewrite(CopyToTensorOp op,
//                                PatternRewriter &rewriter) const override {
//    auto createArrayOp = dyn_cast_or_null<CreateArrayFromTensorOp>(
//        op.getSource().getDefiningOp());
//    if (createArrayOp && createArrayOp.getDest().hasOneUse()) {
//      rewriter.replaceOp(op, createArrayOp.getSource());
//    }
//    return success();
//  }
//};
//} // namespace
//
//void CopyToTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                                 MLIRContext *context) {
//  results.insert<ElideCreateRedundantArrayFromTensor>(context);
//}

#define GET_OP_CLASSES
#include "Dialect/Numpy/IR/NumpyOps.cpp.inc"
