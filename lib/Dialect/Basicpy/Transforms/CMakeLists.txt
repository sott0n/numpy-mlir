add_mlir_conversion_library(ILBasicpyPasses
    TypeInference.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Bascipy/Transforms

    DEPENDS
    ILBasicpyPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    #ILTypingCPA
)