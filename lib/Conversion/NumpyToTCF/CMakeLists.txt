add_mlir_conversion_library(NumpyToTCF
  Passes.cpp

  DEPENDS
  NpcConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  NpcNumpyDialect
  NpcTCF
)
