#include "Dialect/Npcrt/IR/NpcrtDialect.h"
#include "Dialect/Npcrt/IR/NpcrtOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::npc::npcrt;

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

void GlobalOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(this->getName());
  p << " ";
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elidedAttrs=*/{"sym_name", "value"});
  p.printAttribute(this->getValue());
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
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GetGlobalOp::verify() {
  auto global = SymbolTable::lookupNearestSymbolFrom<GlobalOp>(
      *this, this->getGlobalAttr());
  if (!global)
    return this->emitError() << "must reference a valid npcrt.global";
  auto globalType = global.getValue().getType().cast<RankedTensorType>();
  auto resultType = this->getType().cast<ShapedType>();
  if (globalType.getElementType() != resultType.getElementType())
    return this->emitError() << "inconsistent with element type of global";
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleMetadataOp
//===----------------------------------------------------------------------===//

void ModuleMetadataOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  p.printRegion(this->getMetadatas(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult ModuleMetadataOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, std::nullopt, false))
    return failure();
  ModuleMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// FuncMetadataOp
//===----------------------------------------------------------------------===//

LogicalResult FuncMetadataOp::verify() {
  auto *module = (*this)->getParentOp()->getParentOp();
  auto func = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, this->getFuncName()));
  if (!func)
    return this->emitError() << "must reference a valid func";

  if (this->getNumInputs() != func.getNumArguments())
    return this->emitError() << "must agree on number of inputs";
  if (this->getNumOutputs() != func.getNumResults())
    return this->emitError() << "must agree on number of outputs";
  return success();
}

#define GET_OP_CLASSES
#include "Dialect/Npcrt/IR/NpcrtOps.cpp.inc"
