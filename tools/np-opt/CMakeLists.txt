get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIROptLib
  NpInitAll
)

add_llvm_executable(np-opt np-opt.cpp)
llvm_update_compile_flags(np-opt)
target_link_libraries(np-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(np-opt)
