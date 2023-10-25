#ifndef TCP_OPS
#define TCP_OPS

include "Dialect/TCP/IR/TCPBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class TCP_Op<string mnemonic, list<Trait> traits = []> :
        Op<TCP_Dialect, mnemonic, traits>;

// TODO: clarify allowed tensor element types.
// TODO: HasParent is too restrictive? can't have an island with loop.for with
// further ops inside it?
def TCP_AddOp : TCP_Op<"add", []> {
  let summary = "Adds two tensors.";
  let description = [{
    Adds two tensors.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}

def TCP_BroadcastToOp : TCP_Op<"broadcast_to"> {
  let summary = "Broadcasts an operand to a given shape.";
  let description = [{
    Broadcasts `operand` to the shape `shape`.
    It is undefined behavior if such a broadcast is not legal.
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ShapeOrExtentTensorType:$shape);
  let results = (outs AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// Ops that need to be factored to a proper home.
//===----------------------------------------------------------------------===//

// TODO: This probably doesn't belong in the tcp dialect.
def TCP_AllocMemRefOp : TCP_Op<"alloc_memref", []> {
  let summary = "Allocates a memref of the given shape.";
  let description = [{
    Allocates a memref of the given shape.
  }];
  let arguments = (ins Variadic<Shape_ShapeOrExtentTensorType>:$shape);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$shape attr-dict `:` type($shape) `->` type($memref)";
}

def TCP_GlobalOp : TCP_Op<"global", [Symbol]> {
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

def TCP_GetGlobalMemrefOp : TCP_Op<"get_global_memref"> {
  let summary = "Obtain a memref pointing at the given global";
  let description = [{
    Obtain a memref poiting at the given global.
  }];
  let arguments = (ins FlatSymbolRefAttr:$global);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$global attr-dict `:` type($memref)";
  let hasVerifier = 1; 
}

// TODO: Change to a more principled error handling mechanism.
// This op probably doesn't need to exist eventually.
// This op is also not correctly modeled right now, since it itself doesn't
// produce the error in practice. The ops like shape.broadcast itself, when
// lowered, immediately produce errors.
// TODO: This should eventually be moved to a shape dialect.
def TCP_ShapeObserveErrorOp : TCP_Op<"shape_observe_error",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Observes the fact that a shape might be an error.";
  let description = [{
    This op is a structual placeholder that captures a shape such that it
    is not erased. This will keep around shape computations that are later
    lowered into eager error handling code.

    The interaction of this op, especially with control flow and side
    effecting ops, is not very well-defined, and needs to be worked
    or/redesigned.
  }];
  let arguments = (ins Shape_ShapeOrExtentTensorType:$shape);
  // TODO: ODS seems to create redeclared class members if we remove this,
  // resulting in C++ compilation errors.
  let results = (outs NoneType:$dummy);
}

// TODO: This probably belongs in the shape dialect.
def TCP_GetExtentOp : TCP_Op<"get_extent",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gets the specified extent from a shape.";
  let description = [{
    Gets the specified extent from a shape.

    This op has undefined behavior if the shape is an error.
  }];
  let arguments = (ins Shape_ShapeOrExtentTensorType:$shape, I64Attr:$dim);
  let results = (outs Index:$extent);
  let assemblyFormat = "$shape `,` $dim attr-dict `:` type($shape)";

  let builders = [
    // Helper to pass a simple integer instead of an integer attr.
    OpBuilder<(ins "Value":$shape, "int64_t":$dim), [{
      build($_builder, $_state, shape, $_builder.getI64IntegerAttr(dim));
    }]>
  ];
}

#endif // TCP_OPS
