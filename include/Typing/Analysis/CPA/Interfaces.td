#ifndef NPC_TYPING_ANALYSIS_CPA_CPA_INTERFACES
#define NPC_TYPING_ANALYSIS_CPA_CPA_INTERFACES

include "mlir/IR/OpBase.td"

def Npc_TypingCPATypeMapInterface : TypeInterface<
    "NpcTypingTypeMapInterface"> {
  let methods = [
    InterfaceMethod<"Map an MLIR Type to a CPA::TypeNode.",
      /*retTy=*/ "::mlir::npc::Typing::CPA::TypeNode *",
      /*methodName=*/ "mapToCPAType",
      /*args=*/ (ins
        "::mlir::npc::Typing::CPA::Context &":$context)>,
  ];
}

def Npc_CPATypeInferenceOpInterface : OpInterface<
    "NpcTypingCPATypeInferenceOpInterface"> {
  let methods = [
    InterfaceMethod<"Adds CPA constraints for the op.",
      /*retTy=*/ "void",
      /*methodName=*/ "addCPAConstraints",
      /*args=*/ (ins
        "::mlir::npc::Typing::CPA::Context &":$context)>,
  ];
}

#endif // NPC_TYPING_ANALYSIS_CPA_CPA_INTERFACES
