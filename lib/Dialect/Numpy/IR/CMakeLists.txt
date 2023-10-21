add_mlir_dialect_library(ILNumpyDialect
  NumpyDialect.cpp
  NumpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Numpy/IR

  DEPENDS
  NpBasicpyDialect
  MLIRNumpyOpsIncGen
)

target_link_directories(ILNumpyDialect PUBLIC MLIRIR)