#include "Typing/Support/CPAIrHelpers.h"

#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir::npc::Basicpy;

namespace mlir {
namespace npc {
namespace Typing {
namespace CPA {

ObjectValueType::IrTypeConstructor static createTensorLikeIrTypeConstructor(
    TensorType tt) {
  return [tt](ObjectValueType *ovt, llvm::ArrayRef<mlir::Type> fieldTypes,
              MLIRContext *mlirContext,
              std::optional<Location> loc) -> mlir::Type {
    if (auto ranked = tt.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(tt.getShape(), fieldTypes.front());
    } else {
      // Unranked.
      return UnrankedTensorType::get(fieldTypes.front());
    }
  };
}

ObjectValueType *newArrayType(Context &context,
                              ObjectValueType::IrTypeConstructor irCtor,
                              Identifier *typeIdentifier,
                              std::optional<TypeNode *> elementType) {
  TypeNode *concreteElementType;
  if (elementType) {
    concreteElementType = *elementType;
  } else {
    concreteElementType = context.newTypeVar();
  }
  auto arrayElementIdent = context.getIdentifier("e");
  return context.newObjectValueType(irCtor, typeIdentifier, {arrayElementIdent},
                                    {concreteElementType});
}

TypeNode *getArrayElementType(ObjectValueType *arrayType) {
  assert(arrayType->getFieldCount() == 1 &&
         "expected to be an arity 1 array type");
  return arrayType->getFieldTypes().front();
}

ObjectValueType *createTensorLikeArrayType(Context &context,
                                           TensorType tensorType) {
  auto elTy = tensorType.getElementType();
  std::optional<TypeNode *> dtype;
  if (elTy != UnknownType::get(tensorType.getContext())) {
    dtype = context.mapIrType(elTy);
  }
  return newArrayType(context, createTensorLikeIrTypeConstructor(tensorType),
                      context.getIdentifier("!Tensor"), dtype);
}

static TypeNode *defaultTypeMapHook(Context &context, mlir::Type irType) {
  // Handle core types that we can't define an interface on.
  if (auto tensorType = irType.dyn_cast<TensorType>()) {
    return createTensorLikeArrayType(context, tensorType);
  }
  return nullptr;
}

Context::IrTypeMapHook createDefaultTypeMapHook() { return defaultTypeMapHook; }

} // namespace CPA
} // namespace Typing
} // namespace npc
} // namespace mlir