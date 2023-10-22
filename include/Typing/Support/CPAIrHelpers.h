#ifndef NPC_TYPING_SUPPORT_CPA_IR_HELPERS_H
#define NPC_TYPING_SUPPORT_CPA_IR_HELPERS_H

#include "Typing/Analysis/CPA/Types.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace npc {
namespace Typing {
namespace CPA {

/// Creates an array object type with a possibly unknown element type.
/// Be convention, arrays have a single type slot for the element type
/// named 'e'.
ObjectValueType *newArrayType(Context &context,
                              ObjectValueType::IrTypeConstructor irCtor,
                              Identifier *typeIdentifier,
                              std::optional<TypeNode *> elementType);

/// Gets the TypeNode associated with the element type for an array allocated
/// via newArrayType.
TypeNode *getArrayElementType(ObjectValueType *arrayType);

/// Creates an ObjectValueType for the given TensorType. The result will
/// reconstruct the original TensorType's structure but with the resolved
/// element type.
ObjectValueType *createTensorLikeArrayType(Context &context,
                                           TensorType tensorType);

/// Creates a default IR type map hook which supports built-in MLIR types
/// that do not implement the analysis interfaces.
Context::IrTypeMapHook createDefaultTypeMapHook();

} // namespace CPA
} // namespace Typing
} // namespace npc 
} // namespace mlir

#endif // NPC_TYPING_SUPPORT_CPA_IR_HELPERS_H