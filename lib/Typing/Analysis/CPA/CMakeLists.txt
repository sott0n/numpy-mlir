add_library(NpcTypingCPA
  Interfaces.cpp
  Types.cpp
  Algorithm.cpp
)

target_link_libraries(NpcTypingCPA
  MLIRIR
)

add_dependencies(NpcTypingCPA
  NpcTypingCPAInterfacesIncGen
)
