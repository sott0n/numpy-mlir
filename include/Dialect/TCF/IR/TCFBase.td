#ifndef TCF_BASE
#define TCF_BASE

include "mlir/IR/OpBase.td"

def TCF_Dialect : Dialect {
  let name = "tcf";
  let cppNamespace = "::mlir::npc::tcf";
  let description = [{
The `tcf` deialect is a key facilitator for ingesting into the MLIR ecosystem 
dynamic frontend language with a "tensor" primitive type.

Some of its key features are:
- Ops that safely report errors, such as mismatching sizes for a matrix
  multiplication.
- Parameters controlling op behavior are dynamic operands, such as
  convolution window sizes.
- Support for a rank-dynamic programming model.
- Support for implicit broadcasting, following the industry-standard numpy
  broadcasting rules.

These features make this dialect interoperate well with highly-dynamic
programming models as are common in many frontends.

This dialect is optimized for compiler analysis and transformation, especially
lowering to lower levels of abstraction in the compiler.
Tensor programs, as represented in this dialect, are not necessarily represented
in the most efficient way for op-by-op execution.
The goal is that most frontend ops are representable in a small, but
not-necessarily-just-one set of ops from this dialect.
  }];
}

#endif // TCF_BASE