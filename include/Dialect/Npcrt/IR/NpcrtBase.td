#ifndef NPCRT_BASE
#define NPCRT_BASE

include "mlir/IR/OpBase.td"

def Npcrt_Dialect : Dialect {
  let name = "npcrt";
  let cppNamespace = "::mlir::npc::npcrt";
  let description = [{
The `npcrt` dialect is the IR manifestation for interaction with the
il-dialect runtime. It primarily serves as a layer that encapsulates
the data structures and functions available in the runtime, and 
faciliates conversion to these conventions, such as by providing
utilities for being lowered to the llvm dialect.
  }];
  let useDefaultTypePrinterParser = 1;
}

def Npcrt_Tensor
    : DialectType<
          Npcrt_Dialect,
          CPred<"$_self.isa<::mlir::npc::npcrt::TensorType>()">,
          "npcrt.tensor">,
      BuildableType<
          "$_builder.getType<::mlir::npc::npcrt::TensorType>()"> {
  let description = [{The runtime type that represents a buffer.}];
}

#endif // NPCRT_BASE
