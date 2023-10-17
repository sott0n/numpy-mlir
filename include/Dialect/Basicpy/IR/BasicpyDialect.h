#ifndef NPC_DIALECT_BASICPY_IR_BASICPY_DIALECT_H
#define NPC_DIALECT_BASICPY_IR_BASICPY_DIALECT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/Common.h"
//#include "Typing/Analysis/CPA/Interfaces.h"

#include "Dialect/Basicpy/IR/BasicpyOpsDialect.h.inc"

namespace mlir {
namespace npc {
namespace Basicpy {

namespace detail {
struct SlotObjectTypeStorage;
} // namespace detail

namespace BasicpyTypes {
enum Kind {
  // Dialect types.
  BoolType = TypeRanges::Basicpy,
  BytesTypes,
  EllipsisType,
  NoneType,
  SlotObjectType,
  StrType,
  UnknownType,

  // Dialect attributes.
  SingletonAttr,
  LAST_BASICPY_TYPE = SingletonAttr,
};
} // namespace BasicpyTypes

/// Python 'bool' type (can contain values True or False, corresponding to
/// i1 constants of 0 or 1).
class BoolType : public Type::TypeBase<BoolType, Type, TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == BasicpyTypes::BoolType; }
};

/// The type of the Python `bytes` values.
class BytesType : public Type::TypeBase<BytesType, Type, TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == BasicpyTypes::BytesTypes; }
};

/// The type of the Python `Ellipsis` values.
class EllipsisType : public Type::TypeBase<EllipsisType, Type, TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::EllipsisType;
  }
};

/// The type of the Python `None` values.
class NoneType : public Type::TypeBase<NoneType, Type, TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == BasicpyTypes::NoneType; }
};

class SlotObjectType : public Type::TypeBase<SlotObjectType, Type,
                                             detail::SlotObjectTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::SlotObjectType;
  }

  static SlotObjectType get(StringAttr className, ArrayRef<Type> slotTypes);

  StringAttr getClassName();
  unsigned getSlotCount();
  ArrayRef<Type> getSlotTypes();

  // Shorthand to check whether the SlotObject is of a given className and
  // arity.
  bool isOfClassArity(StringRef className, unsigned arity) {
    return getClassName().getValue() == className && getSlotCount() == arity;
  }
};

/// The type of the Python `str` values.
class StrType : public Type::TypeBase<StrType, Type, TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == BasicpyTypes::StrType; }
};

/// An unknown type that could be any supported python type.
class UnknownType : public Type::TypeBase<UnknownType, Type, TypeStorage> {
//                                          IlTypingTypeMapInterface::Trait> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::UnknownType;
  }

  //Typing::CPA::TypeNode *mapToCPAType(Typing::CPA::Context &context);
};

} // namespace Basicpy
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_BASICPY_IR_BASICPY_DIALECT_H
