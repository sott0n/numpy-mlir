get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIROptLib
  NpcInitAll
)

add_llvm_executable(npc-opt npc-opt.cpp)
llvm_update_compile_flags(npc-opt)
target_link_libraries(npc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(npc-opt)
