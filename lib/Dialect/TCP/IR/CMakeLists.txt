add_mlir_dialect_library(NpcTCP
  TCPDialect.cpp
  TCPOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/TCP/IR

  DEPENDS
  MLIRTCPOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShapeDialect
)
