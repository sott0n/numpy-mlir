set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(NpcE2EPassIncGen)

add_mlir_doc(Passes E2EPasses ./ -gen-pass-doc)
