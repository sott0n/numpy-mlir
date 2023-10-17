#include "Dialect/Basicpy/IR/BasicpyOps.h"
#include "Dialect/Basicpy/IR/BasicpyDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/Basicpy/IR/BasicpyOpsEnums.cpp.inc"

namespace mlir {
namespace npc {
namespace Basicpy {

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

//OpFoldResult BoolConstantOp::fold(ArrayRef<Attribute> operands) {
OpFoldResult BoolConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "Bool constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// BytesConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult BytesConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "Bytes constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ExecOp
//===----------------------------------------------------------------------===//

void ExecOp::build(OpBuilder &builder, OperationState &state) {
  OpBuilder::InsertionGuard guard(builder);
  Region *body = state.addRegion();
  builder.createBlock(body);
}

ParseResult ExecOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *bodyRegion = result.addRegion();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  return success();
}

void ExecOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  p.printRegion(this->getBodyRegion());
}

//===----------------------------------------------------------------------===//
// FuncTemplateCallOp
//===----------------------------------------------------------------------===//

LogicalResult FuncTemplateCallOp::verify() {
  auto argNames = this->getArgNames();
  if (argNames.size() > this->getArgs().size()) {
    return this->emitError() << "expected <= kw arg names vs args";
  }

  for (auto it : llvm::enumerate(argNames)) {
    auto argName = it.value().cast<StringAttr>().getValue();
    if (argName == "*" && it.index() != 0) {
      return this->emitError()
             << "positional arg pack must be the first kw arg";
    }
    if (argName == "**" && it.index() != argNames.size() - 1) {
      return this->emitError() << "kw arg pack must be the last kw arg";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FuncTemplateOp
//===----------------------------------------------------------------------===//

void FuncTemplateOp::build(OpBuilder &builder, OperationState &result) {
  OpBuilder::InsertionGuard guard(builder);
  ensureTerminator(*result.addRegion(), builder, result.location);
}

ParseResult FuncTemplateOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *bodyRegion = result.addRegion();
  StringAttr symbolName;

  if (parser.parseSymbolName(symbolName, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  return success();
}

void FuncTemplateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName((*this).getName());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {mlir::SymbolTable::getSymbolAttrName()});
  p << " ";
  p.printRegion(this->getRegion());
}

LogicalResult FuncTemplateOp::verify() {
  Block *body = this->getBody();
  for (auto &childOp : body->getOperations()) {
    if (!llvm::isa<func::FuncOp>(childOp) &&
        !llvm::isa<FuncTemplateTerminatorOp>(childOp)) {
      return childOp.emitError() << "illegal operation in func_template";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SlotObjectMakeOp
//===----------------------------------------------------------------------===//

ParseResult SlotObjectMakeOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operandTypes;
  if (parser.parseOperandList(operandTypes, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseArrowTypeList(result.types)) {
    return failure();
  }

  if (result.types.size() != 1 || !result.types.front().isa<SlotObjectType>()) {
    return parser.emitError(parser.getNameLoc(),
                            "custom assembly form requires Slotobject result");
  }

  auto slotObjectType = result.types.front().cast<SlotObjectType>();
  result.addAttribute("className", slotObjectType.getClassName());
  return parser.resolveOperands(operandTypes, slotObjectType.getSlotTypes(),
                                parser.getNameLoc(), result.operands);
}

void SlotObjectMakeOp::print(OpAsmPrinter &p) {
  // If the argument types do not match the result type slots, then
  // print the generic form.
  auto canCustomPrint = ([&]() -> bool {
    auto type = this->getResult().getType().dyn_cast<SlotObjectType>();
    if (!type)
      return false;
    auto args = this->getSlots();
    auto slotTypes = type.getSlotTypes();
    if (args.size() != slotTypes.size())
      return false;
    for (unsigned i = 0, e = args.size(); i < e; ++i) {
      if (args[i].getType() != slotTypes[i])
        return false;
    }
    return true;
  })();

  if (!canCustomPrint) {
    p.printGenericOp(*this, false);
    return;
  }

  p << "(";
  p.printOperands(this->getSlots());
  p << ")";

  // Not really a symbol but satisfies same rules.
  p.printArrowTypeList(this->getOperation()->getResultTypes());
}

//===----------------------------------------------------------------------===//
// SlotObjectGetOp
//===----------------------------------------------------------------------===//

ParseResult SlotObjectGetOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::UnresolvedOperand object;
  IntegerAttr indexAttr;
  Type indexType = parser.getBuilder().getIndexType();
  if (parser.parseOperand(object) || parser.parseLSquare() ||
      parser.parseAttribute(indexAttr, indexType, "index", result.attributes) ||
      parser.parseRSquare()) {
    return failure();
  }

  Type objectType;
  if (parser.parseColonType(objectType) ||
      parser.resolveOperand(object, objectType, result.operands)) {
    return failure();
  }

  auto castObjectType = objectType.dyn_cast<SlotObjectType>();
  if (!castObjectType) {
    return parser.emitError(parser.getNameLoc(),
                            "illegal object type on custom assembly form");
  }

  auto index = indexAttr.getValue().getZExtValue();
  auto slotTypes = castObjectType.getSlotTypes();
  if (index >= slotTypes.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "out of bound index on custom assembly form");
  }

  result.addTypes({slotTypes[index]});
  return success();
}

void SlotObjectGetOp::print(OpAsmPrinter &p) {
  // If the argument types do not match the result type slots, then
  // print the generic form.
  auto canCustomPrint = ([&]() -> bool {
    auto type = this->getObject().getType().dyn_cast<SlotObjectType>();
    if (!type)
      return false;
    auto index = this->getIndex().getZExtValue();
    if (index >= type.getSlotCount())
      return false;
    if (this->getResult().getType() != type.getSlotTypes()[index])
      return false;
    return true;
  })();

  if (!canCustomPrint) {
    p.printGenericOp(*this, false);
    return;
  }

  p << " ";
  p.printOperand(this->getObject());
  p << "[" << this->getIndex() << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), {"index"});
  p << " : ";
  p.printType(this->getObject().getType());
}

//===----------------------------------------------------------------------===//
// StrConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult StrConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// SingletopOp
//===----------------------------------------------------------------------===//

OpFoldResult SingletonOp::fold(FoldAdaptor adaptor) {
  auto resultType = getResult().getType();
  return TypeAttr::get(resultType);
}

//===----------------------------------------------------------------------===//
// UnknownCastOp
//===----------------------------------------------------------------------===//

namespace {

class ElideIdentityUnknownCast : public OpRewritePattern<UnknownCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnknownCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOperand().getType() != op.getResult().getType())
      return failure();
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

} // namespace

void UnknownCastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.insert<ElideIdentityUnknownCast>(context);
}

} // namespace Basicpy
} // namespace npc
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/Basicpy/IR/BasicpyOps.cpp.inc"
