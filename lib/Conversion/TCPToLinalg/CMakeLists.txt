add_mlir_conversion_library(TCPToLinalg
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/TCPToLinalg

  DEPENDS
  NpcConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShapeDialect
  MLIRLinalgDialect
)
