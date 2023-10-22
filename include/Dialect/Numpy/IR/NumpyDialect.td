#ifndef NPC_DIALECT_NUMPY_IR_NUMPY_DIALECT_H
#define NPC_DIALECT_NUMPY_IR_NUMPY_DIALECT_H

include "mlir/IR/OpBase.td"
include "Dialect/Basicpy/IR/BasicpyDialect.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Numpy_Dialect : Dialect {
  let name = "numpy";
  let summary = "Core numpy dialect";
  let description = [{
      Dialect of types and core numpy ops and abstraction.
  }];
  let cppNamespace = "::mlir::npc::Numpy";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["Basicpy::BasicpyDialect"];
}

//===----------------------------------------------------------------------===//
// Dialect Types
//===----------------------------------------------------------------------===//

def Numpy_AnyDtype : DialectType<Numpy_Dialect,
    CPred<"$_self.isa<::mlir::npc::Numpy::AnyDtypeType>()">,
    "any dtype">,
    BuildableType<"$_builder.getType<::mlir::il::Numpy::AnyDtypeType>()"> {
  let description = [{
    Placeholder for an unknown dtype in a tensor.
  }];
}

def Numpy_NdArrayType : DialectType<Numpy_Dialect,
    CPred<"$_self.isa<::mlir::npc::Numpy::NdArrayType>()">,
    "ndarray type">,
    BuildableType<"$_builder.getType<::mlir::il::Numpy::NdArrayType>()"> {
  let description = [{
    NdArrayType: Models a numpy.ndarray and compatible types.
    Unlike lower level representations, this type solely exists to represent
    top-level semantics and source-dialect transformations. As such, it
    is not a general modeling like `tensor` or `memref`, instead being just
    enough to infer proper lowerings to those types.

    Like its numpy counterparts, NdArrayType represents a mutable array of
    some value type (dtype), with a shape, strides, and various controls
    around contiguity. Most of that is not modeled in this type, which focuses
    on a representation sufficient to infer high level types and aliasing
    based on program flow.

    Note that most operations in numpy can be legally defined similar to the
    following:
      %0 = ... -> !numpy.ndarray<...>
      %1 = numpy.copy_to_tensor %0 -> tensor<...>
      %2 = numpy.some_operation %1
      %4 = numpy.copy_from_tensor -> !numpy.ndarray<...>
    
    (in other words, the operation does not alias any of its operands to its
    results)

    When this is the case, the operation will *only* be defined for tensors,
    as staying in the value domain makes sense for as many operations as
    can be resonably represented as such. It is left to subsequent parts of
    the compiler to transform the program in such a way as to elide the copies
    that such sequences encode.

    Only ops that mutate or alias their operands should accept and/or produce
    ndarray types.
  }];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Any tensor type legal for numpy ops.
def Numpy_AnyTensor : TensorOf<[AnyType]>;

// Any type, at any stage of analysis that can represent a numpy array.
def Numpy_AnyArray : AnyTypeOf<[
  Numpy_AnyTensor,
  Numpy_NdArrayType
]>;

def Numpy_SliceTupleElement : AnyTypeOf<[
  // Supports both "Index Arrays" and "Boolean mask index arrays".
  Numpy_AnyArray,

  // Indicates that an axis should be added (np.newaxis == None).
  Basicpy_NoneType,

  // Indicates that intervening axes should be preserved.
  Basicpy_EllipsisType,

  // A discrete numeric index (represented as IndexType so that a proper
  // width can be target dependent).
  Index,

  // A generalized slice object.
  Basicpy_SliceSlotObjectType,
], "types that are legal elements of a __getitem__ tuple operating on arrays">;

#endif // NPC_DIALECT_NUMPY_IR_NUMPY_DIALECT_H