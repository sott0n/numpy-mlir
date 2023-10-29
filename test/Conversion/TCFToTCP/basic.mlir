// RUN: npc-opt <%s -convert-tcf-to-tcp | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @tcp_add_noshape
func.func @tcp_add_noshape(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // Just the lightest sanity check.
  // CHECK: tcp.add
  %0 = "tcf.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: @tcp_add
func.func @tcp_add(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[N0:.+]] = shape.const_shape [2, 3] : tensor<2xindex>
  // CHECK: %[[N1:.+]] = "tcp.shape_observe_error"(%[[N0]]) : (tensor<2xindex>) -> none
  // CHECK: %[[N2:.+]] = "tcp.broadcast_to"(%arg0, %[[N0]]) : (tensor<2x3xf32>, tensor<2xindex>) -> tensor<2x3xf32>
  // CHECK: %[[N3:.+]] = "tcp.broadcast_to"(%arg1, %[[N0]])
  // CHECK: "tcp.add"(%[[N2]], %[[N3]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = "tcf.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
