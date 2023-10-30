add_mlir_library(NpcE2E
  LowerToHybridTensorMemRef.cpp
  E2E.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/E2E

  DEPENDS
  NpcE2EPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgUtils
)
