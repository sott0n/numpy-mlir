#include "Dialect/TCP/IR/TCPOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::tcp;

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

void GlobalOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName((*this).getName());
  p << " ";
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elidedAttrs=*/{"sym_name", "value"});
  p.printAttribute((*this).getValueAttr());
}

ParseResult GlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Attribute valueAttr;
  if (parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalMemrefOp
//===----------------------------------------------------------------------===//

LogicalResult GetGlobalMemrefOp::verify() {
  auto global = SymbolTable::lookupNearestSymbolFrom<GlobalOp>(
      *this, this->getGlobalAttr());
  if (!global)
    return this->emitError() << "must reference a valid symbol";

  auto globalType = global.getValue().getType().cast<RankedTensorType>();
  auto resultType = this->getType().cast<ShapedType>();
  if (failed(
          verifyCompatibleShape(globalType.getShape(), resultType.getShape())))
    return this->emitError() << "inconsistent with shape of global";

  if (globalType.getElementType() != resultType.getElementType())
    return this->emitError() << "inconsistent with element type of global";
  return success();
}

#define GET_OP_CLASSES
#include "Dialect/TCP/IR/TCPOps.cpp.inc"
