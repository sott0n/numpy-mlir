set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(NpcTCFPassIncGen)

add_mlir_doc(Passes NpcTCFTransforms ./ -gen-pass-doc)