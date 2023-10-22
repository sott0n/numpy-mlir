add_library(NpcTypingCPASupport
  CPAIrHelpers.cpp
)

target_link_libraries(NpcTypingCPASupport
  MLIRIR
  NpcTypingCPA
  NpcBasicpyDialect
)