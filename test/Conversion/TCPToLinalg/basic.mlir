// RUN: npc-opt <%s -convert-tcp-to-linalg | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: tcp_to_linalg
func.func @tcp_to_linalg(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECL: linalg.generic
  %0 = "tcp.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
