#ifndef NPC_DIALECT_COMMON_H
#define NPC_DIALECT_COMMON_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace npc {

namespace TypeRanges {
enum {
  Basicpy = 1,
  Numpy = Basicpy + 50,
  Ilrt = Numpy + 50,
};
}

} // namespace npc
} // namespace mlir

#endif // NPC_DIALECT_COMMON_H
