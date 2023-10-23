add_mlir_conversion_library(BasicpyToStd
  Passes.cpp
  PrimitiveOpsConversion.cpp

  DEPENDS
  NpcConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  NpcBasicpyDialect
)
