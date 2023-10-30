add_mlir_dialect_library(Npcrt
  NpcrtDialect.cpp
  NpcrtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Npcrt/Dialect/Npcrt

  DEPENDS
  MLIRNpcrtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
