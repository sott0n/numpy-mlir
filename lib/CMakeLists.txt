include(NpcCommon)

add_subdirectory(Dialect)
add_subdirectory(Typing)
add_subdirectory(Conversion)
#add_subdirectory(CAPI)
add_subdirectory(E2E)
#add_subdirectory(runtime)
#add_subdirectory(JITRuntime)

#if(IL_ENABLE_REFJIT)
#  add_subdirectory(Backend/RefJIT)
#endif()

add_mlir_library(NpcInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  NpcTCF
  NpcTCP
  NpcE2E
  Npcrt
  NpcBasicpyDialect
  NpcNumpyDialect
  NpcBasicpyPasses
  NpcNumpyPasses
  NpcTCFPasses
  NpcTypingCPASupport
  NpcTypingPasses
  BasicpyToStd
  NumpyToTCF
  TCFToTCP
  TCPToLinalg
)

npc_mlir_target_includes(NpcInitAll)
