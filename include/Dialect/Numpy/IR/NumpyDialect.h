#ifndef NPC_DIALECT_NUMPY_IR_NUMPY_DIALECT_H
#define NPC_DIALECT_NUMPY_IR_NUMPY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/Common.h"
#include "Typing/Analysis/CPA/Interfaces.h"

#include "Dialect/Numpy/IR/NumpyOpsDialect.h.inc"

namespace mlir {
namespace npc {
namespace Numpy {

namespace detail {
struct NdArrayTypeStorage;
} // namespace detail

namespace NumpyTypes {
enum Kind {
  AnyDtypeType = TypeRanges::Numpy,
  NdArray,
  LAST_NUMPY_TYPE = AnyDtypeType,
};
} // namespace NumpyTypes

/// The singleton type representing an unknown dtype.
class AnyDtypeType : public Type::TypeBase<AnyDtypeType, Type, TypeStorage> {
public:
  using Base::Base;

  static AnyDtypeType get(MLIRContext *context) { return Base::get(context); }

  static bool kindof(unsigned kind) { return kind == NumpyTypes::AnyDtypeType; }
};

class NdArrayType
    : public Type::TypeBase<NdArrayType, Type, detail::NdArrayTypeStorage,
                            NpcTypingTypeMapInterface::Trait> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) {
    return kind == NumpyTypes::Kind::NdArray;
  }

  /// Constructs an NdArray with a dtype and no shape. Setting the dtype
  /// to !basicpy.UnknownType will print as ?.
  static NdArrayType
  get(Type dtype, std::optional<ArrayRef<int64_t>> shape = std::nullopt);

  /// Returns whether the dtype is a concrete type (versus
  /// !basicpy.UnknownType).
  bool hasKnownDtype();
  Type getDtype();

  /// If the shape has been partially specified, this will have a value.
  /// unknown dimensions are -1.
  std::optional<ArrayRef<int64_t>> getOptionalShape();

  /// Converts to an equivalent TensorType.
  TensorType toTensorType();

  /// CPA::TypeMapInterface method.
  Typing::CPA::TypeNode *mapToCPAType(Typing::CPA::Context &context);
};

} // namespace Numpy
} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_NUMPY_IR_NUMPY_DIALECT_H
