// RUN: npc-opt -split-input-file -lower-constant-tensors-to-memrefs %s | FileCheck %s --dump-input=fail

// CHECK: module {
// CHECK: tcp.global @__constant_3x4xf32 dense<7.000000e+00> : tensor<3x4xf32>
// CHECK-LABEL: func.func @basic
func.func @basic() -> tensor<3x4xf32> {
  // CHECK: %[[MEMREF:.*]] = tcp.get_global_memref @__constant_3x4xf32 : memref<3x4xf32>
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]] : memref<3x4xf32>
  %0 = arith.constant dense<7.0> : tensor<3x4xf32>
  // CHECK: return %[[TENSOR]]
  return %0 : tensor<3x4xf32>
}
// CHECK: }

// -----

// CHECK: module {
// CHECK: tcp.global
// CHECK-NOT: tcp.global
// CHECK-LABEL: func.func @dupulicate_constants
func.func @dupulicate_constants() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = arith.constant dense<7.0> : tensor<3x4xf32>
  %1 = arith.constant dense<7.0> : tensor<3x4xf32>
  // CHECK: return
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}
// CHECK: }

// -----

// CHECK: module {
// CHECK: tcp.global
// CHECK: tcp.global
// CHECK-NOT: tcp.global
// CHECK-LABEL: func.func @multiple_constants
func.func @multiple_constants() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = arith.constant dense<7.0> : tensor<3x4xf32>
  %1 = arith.constant dense<8.0> : tensor<3x4xf32>
  // CHECK: return
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}
// CHECK: }

// -----

// CHECK: module {
// Don't convert non-tensor globals.
// CHECK-NOT: tcp.global
func.func @non_tensor() {
  %0 = arith.constant 7 : i32
  return
}
// CHECK: }
