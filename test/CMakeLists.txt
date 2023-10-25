configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPC_TEST_DEPENDS
  FileCheck count not
  npc-opt
  #il-run-mlir
)

add_lit_testsuite(check-npc "Running the numpy-compiler's regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NPC_TEST_DEPENDS}
)
set_target_properties(check-npc PROPERTIES FOLDER "Tests")

add_lit_testsuites(NPC ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NPC_TEST_DEPENDS})
