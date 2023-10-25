add_mlir_conversion_library(NpcTypingPasses
  CPATypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Typing/Transforms

  DEPENDS
  NpcTypingTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NpcTypingCPA
  NpcTypingCPASupport
)
