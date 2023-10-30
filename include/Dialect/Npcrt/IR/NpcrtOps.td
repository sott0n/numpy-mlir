#ifndef NPCRT_OPS
#define NPCRT_OPS

include "Dialect/Npcrt/IR/NpcrtBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Npcrt_Op<string mnemonic, list<Trait> traits = []> :
        Op<Npcrt_Dialect, mnemonic, traits>;

def Npcrt_ToMemrefOp : Npcrt_Op<"to_memref"> {
  let summary = "Gets a memref descripter from a tensor";
  let description = [{
    Gets a memref descriptor from a tensor.
  }];
  let arguments = (ins Npcrt_Tensor:$tensor);
  let results = (outs AnyUnrankedMemRef:$memref);
  let assemblyFormat = "$tensor attr-dict `:` type($memref)";
}

def Npcrt_FromMemrefOp : Npcrt_Op<"from_memref"> {
  let summary = "Converts a memref descriptor to a tensor";
  let description = [{
    Copies the data from memref into a new tensor.
  }];
  let arguments = (ins AnyUnrankedMemRef:$memref);
  let results = (outs Npcrt_Tensor:$tensor);
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

def Npcrt_GetExtentOp : Npcrt_Op<"get_extent"> {
  let summary = "Gets the specified extent of the tensor";
  let description = [{
    Gets the `dim`'th extent of the tensor.
  }];
  let arguments = (ins Npcrt_Tensor:$tensor, I32:$dim);
  let results = (outs Index:$extent);
  let assemblyFormat = "$tensor `,` $dim attr-dict";
}

def Npcrt_AbortIfOp : Npcrt_Op<"abort_if"> {
  let summary = "Aborts if the predicate is true";
  let description = [{
    Aborts if the predicate is true.
  }];
  let arguments = (ins I1:$pred);
  let results = (outs);
  let assemblyFormat = "$pred attr-dict";
}

def Npcrt_GlobalOp : Npcrt_Op<"global", [Symbol]> {
  let summary = "Represents a global variable";
  let description = [{
    Represents a global variable.

    Currently, only constant tensors are supported, and they are not
    considered to be exported.
  }];
  let arguments = (ins StrAttr:$sym_name, ElementsAttr:$value);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
}

def Npcrt_GetGlobalOp : Npcrt_Op<"get_global"> {
  let summary = "Obtain a rank-erased memref pointing at the given global";
  let description = [{
    Obtain a rank-erased memref pointing at the given global.
  }];
  let arguments = (ins FlatSymbolRefAttr:$global);
  let results = (outs AnyUnrankedMemRef:$memref);
  let assemblyFormat = "$global attr-dict `:` type($memref)";
  let hasVerifier = 1;
}

def Npcrt_ModuleMetadataOp : Npcrt_Op<"module_metadata", [
  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
]> {
  let summary = "Global metadata for the module.";
  let description = [{
    This op contains a region containing Npcrt.func_metadata ops,
    which give information about the function in the module. This allows
    the module to be introspected when it is loaded, such as looking up
    functions.
    Future uses are checking how many results functions should have, or
    what their argument types are expected to be to provide clean and safe
    errors when invocations faNpc.

    This op is desinged to hold a region, which makes it easy to convert to
    a single LLVM global with a single conversion pattern.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$metadatas);
  let hasCustomAssemblyFormat = 1;
}

def Npcrt_ModuleMetadataTerminatorOp
      : Npcrt_Op<"module_metadata_terminator",
                [Terminator, HasParent<"ModuleMetadataOp">]> {
  let summary = "Implicit terminator for ModuleMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Npcrt_FuncMetadataOp
      : Npcrt_Op<"func_metadata", [HasParent<"ModuleMetadataOp">]> {
  let summary = "Runtime metadata for a single func";
  let description = [{
    Runtime metadata for a single func.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$funcName,
    I32Attr:$numInputs,
    I32Attr:$numOutputs
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // NPCRT_OPS
