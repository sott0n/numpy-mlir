#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Npcrt/IR/NpcrtDialect.h"
#include "Dialect/Npcrt/IR/NpcrtOps.h"

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::npcrt;

#include "Dialect/Npcrt/IR/NpcrtOpsDialect.cpp.inc"

Type NpcrtDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "tensor")
    return TensorType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type in 'ilrt' dialect: ")
      << keyword;
  return Type();
}

void NpcrtDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<npcrt::TensorType>([&](Type) {
        os << "tensor";
        return;
      })
      .Default([](Type) { llvm_unreachable("unexpected npcrt's type kind"); });
}

void NpcrtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Npcrt/IR/NpcrtOps.cpp.inc"
      >();
  addTypes<TensorType>();
}
