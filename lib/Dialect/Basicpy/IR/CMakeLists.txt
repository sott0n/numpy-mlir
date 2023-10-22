add_mlir_dialect_library(NpcBasicpyDialect
  BasicpyDialect.cpp
  BasicpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Basicpy/IR

  DEPENDS
  MLIRBasicpyOpsIncGen
  NpcTypingCPA
)

target_link_directories(NpcBasicpyDialect PUBLIC MLIRIR)