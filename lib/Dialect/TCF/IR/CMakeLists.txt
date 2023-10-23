add_mlir_dialect_library(NpcTCF
  TCFDialect.cpp
  TCFOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/TCF/IR

  DEPENDS
  MLIRTCFOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
