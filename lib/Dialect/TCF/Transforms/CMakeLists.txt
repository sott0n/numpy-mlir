add_mlir_conversion_library(NpcTCFPasses
  ShapeRefinement.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/TCF/Transforms

  DEPENDS
  NpcTCFPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NpcTCF
)
