#ifndef TCP_BASE
#define TCP_BASE

include "mlir/IR/OpBase.td"

def TCP_Dialect : Dialect {
  let name = "tcp";
  let cppNamespace = "::mlir::npc::tcp";
  let description = [{
The `tcp` dialect is the gateway to MLIR's code generation infrastructure.
It is also a great place to do algebraic transformations making use of
semantically-changed names ops.

Features:
- Requires ranked tensors (except for a handful a special ops).
- No implicit broadcasting.
- Performance-critical parameters like convolution window sizes are represented
  with attributes.
- Attention to detail modeling ops that are logically "pure" but have
  preconditions.

Together these features allow a relatively large class of "common-sense"
optimizations to be done with only modestly complex considerations.
// TODO: consider having these ops take a "witness" argument
// that makes them truly NoSideEffect?
// Or have a totally pure "tcp.island" op?
// Figure it out when doing the tcf to tcp lowering.
  }];
}

#endif // TCP_BASE
