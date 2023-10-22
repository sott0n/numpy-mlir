add_mlir_dialect_library(NpcNumpyDialect
  NumpyDialect.cpp
  NumpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Numpy/IR

  DEPENDS
  NpcBasicpyDialect
  MLIRNumpyOpsIncGen
)

target_link_directories(NpcNumpyDialect PUBLIC MLIRIR)