add_mlir_conversion_library(NpcNumpyPasses
    PublicFunctionsToTensor.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Numpy/Transforms

    DEPENDS
    NpcNumpyPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    NpcNumpyDialect
)