#ifndef NPC_DIALECT_BASICPY_IR_BASICPY_DIALECT
#define NPC_DIALECT_BASICPY_IR_BASICPY_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Basicpy_Dialect : Dialect {
  let name = "basicpy";
  let summary = "Basic Python dialect";
  let description = [{
    Core types and ops of Basicpy.
  }];
  let cppNamespace = "::mlir::npc::Basicpy";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//

class Basicpy_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Basicpy_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Basicpy_BoolType : Basicpy_Type<"Bool", "bool"> {
  let description = [{
    Type for 'True' and 'False' values.
  }];
}

def Basicpy_BytesType : Basicpy_Type<"Bytes", "bytes"> {
  let description = [{
    Represents Python 'bytes' values.
  }];
}

def Basicpy_EllipsisType : Basicpy_Type<"Ellipsis", "ellipsis"> {
  let description = [{
    Type of the Python 'Ellipsis' values.
  }];
}

def Basicpy_NoneType : Basicpy_Type<"None", "None"> {
  let description = [{
    Type of the Python 'None' values.
  }];
}

def Basicpy_SlotObjectType : Basicpy_Type<"SlotObject", "slotObject"> {
  let description = [{
    Type for built-in objects which have a fixed number of slots and a type
    name in the system catalog of types. In some ways, this resemblesa a
    namedtuple, but it is used for built-in custom objects.
  }];

  let builders = [
    TypeBuilder<(ins "StringAttr":$className, "ArrayRef<Type>":$slotTypes), [{
      return Base::get(className.getContext(), BasicpyTypes::SlotObjectType,
                       className, slotTypes);
    }]>
  ];
}

def Basicpy_StrType : Basicpy_Type<"Str", "str"> {
  let description = [{
    Representaion values of the python 'str' type.
  }];
}

def Basicpy_UnknownType : Basicpy_Type<"Unknown", "unknown"> {
  let description = [{
    A unknown type (for the current phase of processing).
  }];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

def Basicpy_SingletonType : AnyTypeOf<[
  Basicpy_NoneType,
  Basicpy_EllipsisType,
]>;

// A predicate to determine whether a Type is a SlotObject of a given
// className and arity. Does no checking of slot types.
class Basicpy_SlotObjectOfClassArity<string className, int arity> :
  And<[
    Basicpy_SlotObjectType.predicate,
    CPred<
      "$_self.cast<::mlir::npc::Basicpy::SlotObjectType>().isOfClassArity(\""
      # className # "\", " # arity # ")">
  ]>;

// Type representing a 'slice' object, which mirrors the python built-in
// slice class.
def Basicpy_SliceSlotObjectType : 
  Type<Basicpy_SlotObjectOfClassArity<"slice", 3>>;


#endif // NPC_DIALECT_BASICPY_IR_BASICPY_DIALECT
