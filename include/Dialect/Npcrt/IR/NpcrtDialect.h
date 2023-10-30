#ifndef NPC_DIALECT_NPCILT_IR_NPCRTDIALECT_H
#define NPC_DIALECT_NPCILT_IR_NPCRTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "Dialect/Common.h"

namespace mlir {
namespace npc {
namespace npcrt {

namespace NpcrtTypes {
enum Kind { TensorType = TypeRanges::Npcrt };
} // namespace NpcrtTypes

class TensorType : public Type::TypeBase<TensorType, Type, TypeStorage> {
public:
  using Base::Base;

  static TensorType get(MLIRContext *context) { return Base::get(context); }

  static bool kindof(unsigned kind) {
    return kind == NpcrtTypes::Kind::TensorType;
  }
};

} // namespace npcrt
} // namespace npc
} // namespace mlir

#include "Dialect/Npcrt/IR/NpcrtOpsDialect.h.inc"

#endif // NPC_DIALECT_NPCRT_IR_NPCRTDIALECT_H
