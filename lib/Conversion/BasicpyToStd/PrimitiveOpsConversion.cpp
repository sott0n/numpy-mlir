#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#include "Conversion/BasicpyToStd/Patterns.h"
#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include "Dialect/Basicpy/IR/BasicpyOps.h"

using namespace mlir;
using namespace npc;

namespace {

bool isLegalBinaryOpType(Type type) {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth() > 1; // Do not match i1
  }
  return false;
}

// Convert to std ops when all types match. It is assumed that additional
// patterns and type inference are used to get into this form.
class NumericBinaryExpr : public OpRewritePattern<Basicpy::BinaryExprOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Basicpy::BinaryExprOp op,
                                PatternRewriter &rewriter) const override {
    // Match failure unless if both:
    //   a) operands/results are the same type
    //   b) matches a set of supported primitive types
    //   c) the operation maps to a simple std op without further massaging
    auto valueType = op.getLeft().getType();
    if (valueType != op.getRight().getType() ||
        valueType != op.getResult().getType())
      return failure();

    if (!isLegalBinaryOpType(valueType))
      return failure();

    auto operation = Basicpy::symbolizeBinaryOperation(
        Basicpy::stringifyBinaryOperation(op.getOps()));
    if (!operation)
      return failure();

    auto left = op.getLeft();
    auto right = op.getRight();

    // Generally, int and float ops in std are different.
    using Basicpy::BinaryOperation;
    if (valueType.isa<IntegerType>()) {
      // Note that not all operation make sense or are defined for integer
      // math. Of specific note is the Div vs. FloorDiv distinction.
      switch (*operation) {
      case BinaryOperation::Add:
        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, left, right);
        return success();
      case BinaryOperation::BitAnd:
        rewriter.replaceOpWithNewOp<arith::AndIOp>(op, left, right);
        return success();
      case BinaryOperation::BitOr:
        rewriter.replaceOpWithNewOp<arith::OrIOp>(op, left, right);
        return success();
      case BinaryOperation::BitXor:
        rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, left, right);
        return success();
      case BinaryOperation::FloorDiv:
        // TODO: This is not a precise match for negative division.
        // SignedDivOp rounds towards zero and python rounds towards
        // most negative.
        rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, left, right);
        return success();
      case BinaryOperation::LShift:
        rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, left, right);
        return success();
      case BinaryOperation::Mod:
        rewriter.replaceOpWithNewOp<arith::RemSIOp>(op, left, right);
        return success();
      case BinaryOperation::Mult:
        rewriter.replaceOpWithNewOp<arith::MulIOp>(op, left, right);
        return success();
      case BinaryOperation::RShift:
        rewriter.replaceOpWithNewOp<arith::ShRSIOp>(op, left, right);
        return success();
      case BinaryOperation::Sub:
        rewriter.replaceOpWithNewOp<arith::SubIOp>(op, left, right);
        return success();
      default:
        return failure();
      }
    } else if (valueType.isa<FloatType>()) {
      // Note that most operations are not supported on floating point value.
      // In addition, some cannot be directly implemented with single std
      // ops.
      switch (*operation) {
      case BinaryOperation::Add:
        rewriter.replaceOpWithNewOp<arith::AddFOp>(op, left, right);
        return success();
      case BinaryOperation::Div:
        rewriter.replaceOpWithNewOp<arith::DivFOp>(op, left, right);
        return success();
      case BinaryOperation::FloorDiv:
        // TODO: Implement floating point floor division.
        return rewriter.notifyMatchFailure(
            op, "floating point floor division not implemented");
      case BinaryOperation::Mod:
        // TODO: Implement floating point mod.
        return rewriter.notifyMatchFailure(
            op, "floating point mod not implemented");
      case BinaryOperation::Mult:
        rewriter.replaceOpWithNewOp<arith::MulFOp>(op, left, right);
        return success();
      case BinaryOperation::Sub:
        rewriter.replaceOpWithNewOp<arith::SubFOp>(op, left, right);
        return success();
      default:
        return failure();
      }
    }
    return failure();
  }
};

std::optional<arith::CmpIPredicate>
mapBasicpyPredicateToCmpI(Basicpy::CompareOperation predicate) {
  using Basicpy::CompareOperation;
  switch (predicate) {
  case CompareOperation::Eq:
    return arith::CmpIPredicate::eq;
  case CompareOperation::Gt:
    return arith::CmpIPredicate::sgt;
  case CompareOperation::GtE:
    return arith::CmpIPredicate::sge;
  case CompareOperation::Is:
    return arith::CmpIPredicate::eq;
  case CompareOperation::IsNot:
    return arith::CmpIPredicate::ne;
  case CompareOperation::Lt:
    return arith::CmpIPredicate::slt;
  case CompareOperation::LtE:
    return arith::CmpIPredicate::sle;
  case CompareOperation::NotEq:
    return arith::CmpIPredicate::ne;
  default:
    return std::nullopt;
  }
}

std::optional<arith::CmpFPredicate>
mapBasicpyPredicateToCmpF(Basicpy::CompareOperation predicate) {
  using Basicpy::CompareOperation;
  switch (predicate) {
  case CompareOperation::Eq:
    return arith::CmpFPredicate::OEQ;
  case CompareOperation::Gt:
    return arith::CmpFPredicate::OGT;
  case CompareOperation::GtE:
    return arith::CmpFPredicate::OGE;
  case CompareOperation::Is:
    return arith::CmpFPredicate::OEQ;
  case CompareOperation::IsNot:
    return arith::CmpFPredicate::ONE;
  case CompareOperation::Lt:
    return arith::CmpFPredicate::OLT;
  case CompareOperation::LtE:
    return arith::CmpFPredicate::OLE;
  case CompareOperation::NotEq:
    return arith::CmpFPredicate::ONE;
  default:
    return std::nullopt;
  }
}

class NumericCompare : public OpRewritePattern<Basicpy::BinaryCompareOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Basicpy::BinaryCompareOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto valueType = op.getLeft().getType();
    if (valueType != op.getRight().getType())
      return failure();

    if (!isLegalBinaryOpType(valueType))
      return failure();

    auto caseValueStr = stringifyCompareOperation(op.getOps());
    auto bpyPredicate = Basicpy::symbolizeCompareOperation(caseValueStr);
    if (!bpyPredicate) {
      return failure();
    }

    if (valueType.isa<IntegerType>()) {
      if (auto stdPredicate = mapBasicpyPredicateToCmpI(*bpyPredicate)) {
        auto cmp = rewriter.create<arith::CmpIOp>(loc, *stdPredicate,
                                                  op.getLeft(), op.getRight());
        rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(
            op, Basicpy::BoolType::get(rewriter.getContext()), cmp);
        return success();
      } else {
        return rewriter.notifyMatchFailure(
            op, "unsupported compare operation of integer");
      }
    } else if (valueType.isa<FloatType>()) {
      if (auto stdPredicate = mapBasicpyPredicateToCmpF(*bpyPredicate)) {
        auto cmp = rewriter.create<arith::CmpFOp>(loc, *stdPredicate,
                                                  op.getLeft(), op.getRight());
        rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(
            op, Basicpy::BoolType::get(rewriter.getContext()), cmp);
        return success();
      } else {
        return rewriter.notifyMatchFailure(
            op, "unsupprted compare operation of floating point");
      }
    }
    return failure();
  }
};

// Converts the to_boolean op for numeric types.
class NumericToBoolean : public OpRewritePattern<Basicpy::ToBooleanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Basicpy::ToBooleanOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto operandType = op->getOperand(0).getType();
    if (operandType.isa<IntegerType>()) {
      auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, operandType);
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ne,
                                                 op.getOperand(), zero);
      return success();
    } else if (operandType.isa<FloatType>()) {
      auto zero = rewriter.create<arith::ConstantOp>(
          loc, operandType, FloatAttr::get(operandType, 0.0));
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::ONE,
                                                 op.getOperand(), zero);
      return success();
    }
    return failure();
  }
};

} // namespace

void mlir::npc::populateBasicpyToStdPrimitiveOpPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.insert<NumericBinaryExpr>(context);
  patterns.insert<NumericCompare>(context);
  patterns.insert<NumericToBoolean>(context);
}
