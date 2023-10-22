add_mlir_conversion_library(NpcBasicpyPasses
    TypeInference.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Bascipy/Transforms

    DEPENDS
    NpcBasicpyPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    NpcTypingCPA
)