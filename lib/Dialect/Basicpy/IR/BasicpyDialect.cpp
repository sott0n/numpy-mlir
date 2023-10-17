#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include "Dialect/Basicpy/IR/BasicpyOps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Basicpy/IR/BasicpyOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::Basicpy;

Type BasicpyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "BoolType")
    return BoolType::get(getContext());
  //if (keyword == "BytesType")
  //  return BytesType::get(getContext());
  //if (keyword == "EllipsisType")
  //  return EllipsisType::get(getContext());
  //if (keyword == "NoneType")
  //  return NoneType::get(getContext());
  //if (keyword == "SlotObject") {
  //  StringRef className;
  //  if (parser.parseLess() || parser.parseKeyword(&className)) {
  //    return Type();
  //  }
  //  llvm::SmallVector<Type, 4> slotTypes;
  //  while (succeeded(parser.parseOptionalComma())) {
  //    Type slotType;
  //    if (parser.parseType(slotType))
  //      return Type();
  //    slotTypes.push_back(slotType);
  //  }
  //  if (parser.parseGreater())
  //    return Type();

  //  return SlotObjectType::get(StringAttr::get(getContext(), className),
  //                             slotTypes);
  //}
  //if (keyword == "StrType")
  //  return StrType::get(getContext());
  //if (keyword == "UnknownType")
  //  return UnknownType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type in 'basicpy' dialect: ")
      << keyword;
  return Type();
}

void BasicpyDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<Basicpy::BoolType>([&](Type) { os << "BoolType"; })
      //.Case<Basicpy::BytesType>([&](Type) { os << "BytesType"; })
      //.Case<Basicpy::EllipsisType>([&](Type) { os << "EllipsisType"; })
      //.Case<Basicpy::NoneType>([&](Type) { os << "NoneType"; })
      //.Case<Basicpy::SlotObjectType>([&](Type) {
      //  auto slotObject = type.cast<SlotObjectType>();
      //  auto slotTypes = slotObject.getSlotTypes();
      //  os << "SlotObject<" << slotObject.getClassName().getValue();
      //  if (!slotTypes.empty()) {
      //    os << ", ";
      //    llvm::interleaveComma(slotTypes, os,
      //                          [&](Type t) { os.printType(t); });
      //  }
      //  os << ">";
      //})
      //.Case<Basicpy::StrType>([&](Type) { os << "StrType"; })
      //.Case<Basicpy::UnknownType>([&](Type) { os << "UnknownType"; })
      .Default(
          [](Type) { llvm_unreachable("Unexpected Basicpy's type kind"); });
}

//----------------------------------------------------------------------------//
// Type and attribute detail
//----------------------------------------------------------------------------//

//namespace mlir {
//namespace npc {
//namespace Basicpy {
//namespace detail {
//
//struct SlotObjectTypeStorage : public TypeStorage {
//  using KeyTy = std::pair<StringAttr, ArrayRef<Type>>;
//  SlotObjectTypeStorage(StringAttr className, ArrayRef<Type> slotTypes)
//      : className(className), slotTypes(slotTypes) {}
//
//  bool operator==(const KeyTy &other) const {
//    return className == other.first && slotTypes == other.second;
//  }
//
//  static llvm::hash_code hashKey(const KeyTy &key) {
//    return llvm::hash_combine(key.first, key.second);
//  }
//
//  static SlotObjectTypeStorage *construct(TypeStorageAllocator &allocator,
//                                          const KeyTy &key) {
//    ArrayRef<Type> slotTypes = allocator.copyInto(key.second);
//    return new (allocator.allocate<SlotObjectTypeStorage>())
//        SlotObjectTypeStorage(key.first, slotTypes);
//  }
//
//  StringAttr className;
//  ArrayRef<Type> slotTypes;
//};
//
//} // namespace detail
//} // namespace Basicpy
//} // namespace il
//} // namespace mlir
//
//StringAttr SlotObjectType::getClassName() { return getImpl()->className; }
//ArrayRef<Type> SlotObjectType::getSlotTypes() { return getImpl()->slotTypes; }
//unsigned SlotObjectType::getSlotCount() { return getImpl()->slotTypes.size(); }
//
//SlotObjectType SlotObjectType::get(StringAttr className,
//                                   ArrayRef<Type> slotTypes) {
//  return Base::get(className.getContext(), className, slotTypes);
//}
//
////----------------------------------------------------------------------------//
//// CPA Interface Implementation
////----------------------------------------------------------------------------//
//
//Typing::CPA::TypeNode *
//UnknownType::mapToCPAType(Typing::CPA::Context &context) {
//  return context.newTypeVar();
//}

void BasicpyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Basicpy/IR/BasicpyOps.cpp.inc"
      >();
  //addTypes<BoolType, BytesType, EllipsisType, NoneType, SlotObjectType, StrType,
  //         UnknownType>();
  addTypes<BoolType>();

  // TODO: Make real ops for everything we need.
  allowUnknownOperations();
}
