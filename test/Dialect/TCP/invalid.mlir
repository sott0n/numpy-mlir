// RUN: npc-opt -split-input-file %s -verify-diagnostics

// -----
tcp.global @g dense<0.0> : tensor<2xf32>
func.func @invalid_symbol() {
  // expected-error @+1 {{must reference a valid symbol}}
  tcp.get_global_memref @nonexistent_symbol : memref<3xf32>
  return
}

// -----
tcp.global @g dense<0.0> : tensor<2xf32>
func.func @inconsistent_shape() {
  // expected-error @+1 {{inconsistent with shape of global}}
  tcp.get_global_memref @g : memref<3xf32>
  return
}

// -----
tcp.global @g dense<0.0> : tensor<2xf32>
func.func @inconsistent_element_type() {
  // expected-error @+1 {{inconsistent with element type of global}}
  tcp.get_global_memref @g : memref<2xi8>
  return
}
