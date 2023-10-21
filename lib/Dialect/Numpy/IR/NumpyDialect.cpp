#include "Dialect/Numpy/IR/NumpyDialect.h"
#include "Dialect/Numpy/IR/NumpyOps.h"
#include "Dialect/Basicpy/IR/BasicpyDialect.h"
//#include "Typing/Support/CPAIrHelpers.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/LoopInfo.h"

#include "Dialect/Numpy/IR/NumpyOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::Numpy;

Type NumpyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "any_dtype")
    return AnyDtypeType::get(getContext());
  if (keyword == "ndarray") {
    // Parse:
    //   ndarray<*:?>
    //   ndarray<*:i32>
    //   ndarray<[1,2,3]:i32>
    //
    // Note that this is a different syntax than the built-ins as the dialect
    // parser is not general enough to parse a dimension list with an optional
    // element type (?). The built-in form is also remarkably ambigous when
    // considering extending it.
    Type dtype = Basicpy::UnknownType::get(getContext());
    bool hasShape = false;
    llvm::SmallVector<int64_t, 4> shape;

    if (parser.parseLess())
      return Type();
    if (succeeded(parser.parseOptionalStar())) {
      // Unranked
    } else {
      // Parse dimension list.
      hasShape = true;
      if (parser.parseLSquare())
        return Type();
      for (bool first = true;; first = false) {
        if (!first) {
          if (failed(parser.parseOptionalComma())) {
            break;
          }
        }
        if (succeeded(parser.parseOptionalQuestion())) {
          shape.push_back(ShapedType::kDynamic);
          continue;
        }

        int64_t dim;
        auto optionalPr = parser.parseOptionalInteger(dim);
        if (optionalPr.has_value()) {
          if (failed(*optionalPr))
            return Type();
          shape.push_back(dim);
          continue;
        }
        break;
      }
      if (parser.parseRSquare()) {
        return Type();
      }
    }
    // Parse colon dtype.
    if (parser.parseColon()) {
      return Type();
    }

    if (failed(parser.parseOptionalQuestion())) {
      // Specified dtype.
      if (parser.parseType(dtype)) {
        return Type();
      }
    }
    if (parser.parseGreater()) {
      return Type();
    }

    std::optional<ArrayRef<int64_t>> optionalShape;
    if (hasShape)
      optionalShape = shape;
    auto ndarray = NdArrayType::get(dtype, optionalShape);
    return ndarray;
  }

  parser.emitError(parser.getNameLoc(), "unknown type in 'Numpy' dialect: ")
      << keyword;
  return Type();
}

void NumpyDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<Numpy::AnyDtypeType>([&](Type) { os << "any_dtype"; })
      .Case<Numpy::NdArrayType>([&](Type) {
        auto unknownType = Basicpy::UnknownType::get(getContext());
        auto ndarray = type.cast<NdArrayType>();
        auto shape = ndarray.getOptionalShape();
        auto dtype = ndarray.getDtype();
        os << "ndarray<";
        if (!shape) {
          os << "*:";
        } else {
          os << "[";
          for (auto it : llvm::enumerate(*shape)) {
            if (it.index() > 0)
              os << ",";
            if (it.value() < 0)
              os << "?";
            else
              os << it.value();
          }
          os << "]:";
        }
        if (dtype != unknownType)
          os.printType(dtype);
        else
          os << "?";
        os << ">";
      })
      .Default([](Type) { llvm_unreachable("Unexpected Numpy's type kind"); });
}

//----------------------------------------------------------------------------//
// Type and attribute detail
//----------------------------------------------------------------------------//

namespace mlir {
namespace npc {
namespace Numpy {
namespace detail {

struct NdArrayTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, std::optional<ArrayRef<int64_t>>>;
  NdArrayTypeStorage(Type dtype, int rank, const int64_t *shapeElements)
      : dtype(dtype), rank(rank), shapeElements(shapeElements) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dtype, getOptionalShape());
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.second) {
      return llvm::hash_combine(key.first, *key.second);
    } else {
      return llvm::hash_combine(key.first, -1);
    }
  }

  static NdArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    int rank = -1;
    const int64_t *shapeElements = nullptr;
    if (key.second.has_value()) {
      auto allocElements = allocator.copyInto(*key.second);
      rank = key.second->size();
      shapeElements = allocElements.data();
    }
    return new (allocator.allocate<NdArrayTypeStorage>())
        NdArrayTypeStorage(key.first, rank, shapeElements);
  }

  std::optional<ArrayRef<int64_t>> getOptionalShape() const {
    if (rank < 0)
      return std::nullopt;
    return ArrayRef<int64_t>(shapeElements, rank);
  }

  Type dtype;
  int rank;
  const int64_t *shapeElements;
};

} // namespace detail
} // namespace Numpy
} // namespace npc
} // namespace mlir

NdArrayType NdArrayType::get(Type dtype,
                             std::optional<ArrayRef<int64_t>> shape) {
  assert(dtype && "dtype cannot be null");
  return Base::get(dtype.getContext(), dtype, shape);
}

bool NdArrayType::hasKnownDtype() {
  return getDtype() != Basicpy::UnknownType::get(getContext());
}

Type NdArrayType::getDtype() { return getImpl()->dtype; }

std::optional<ArrayRef<int64_t>> NdArrayType::getOptionalShape() {
  return getImpl()->getOptionalShape();
}

TensorType NdArrayType::toTensorType() {
  auto shape = getOptionalShape();
  if (shape) {
    return RankedTensorType::get(*shape, getDtype());
  } else {
    return UnrankedTensorType::get(getDtype());
  }
}

//Typing::CPA::TypeNode *
//NdArrayType::mapToCPAType(Typing::CPA::Context &context) {
//  std::optional<Typing::CPA::TypeNode *> dtype;
//  if (hasKnownDtype()) {
//    // TODO: This should be using a general mechanism for resolving the dtype.
//    // but we don't have that yet, and for NdArray, these must be primitives
//    // anyway.
//    dtype = context.getIRValueType(getDtype());
//  }
//  // Safe to capture an ArrayRef backed by type storage since it is uniqued.
//  auto optionalShape = getOptionalShape();
//  auto irCtor = [optionalShape](Typing::CPA::ObjectValueType *ovt,
//                                llvm::ArrayRef<mlir::Type> fieldTypes,
//                                MLIRContext *mlirContext,
//                                std::optional<Location>) {
//    assert(fieldTypes.size() == 1);
//    return NdArrayType::get(fieldTypes.front(), optionalShape);
//  };
//  return Typing::CPA::newArrayType(context, irCtor,
//                                   context.getIdentifier("!NdArray"), dtype);
//}

void NumpyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Numpy/IR/NumpyOps.cpp.inc"
      >();
  addTypes<AnyDtypeType, NdArrayType>();
}
