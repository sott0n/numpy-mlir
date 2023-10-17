add_mlir_dialect_library(NpBasicpyDialect
  BasicpyDialect.cpp
  BasicpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Basicpy/IR

  DEPENDS
  MLIRBasicpyOpsIncGen
  #ILTypingCPA
)

target_link_directories(NpBasicpyDialect PUBLIC MLIRIR)