// RUN: npc-opt %s -lower-alloc-memref-ops | FileCheck %s --dump-input=fail

// CHECK-LABEL: func.func @basic
func.func @basic(%arg0: !shape.shape) {
  // CHECK: %[[E:.*]] = tcp.get_extent %arg0, 0
  // CHECK: memref.alloc(%[[E]])
  %0 = tcp.alloc_memref %arg0 : !shape.shape -> memref<?xf32>
  return
}

// CHECK-LABEL: func.func @all_static
func.func @all_static(%arg0: !shape.shape) {
  // CHECK-NOT: tcp.get_extent
  // CHECK: memref.alloc()
  %0 = tcp.alloc_memref %arg0 : !shape.shape -> memref<3x4x5xf32>
  return
}

// CHECK-LABEL: func.func @some_static
func.func @some_static(%arg0: !shape.shape) {
  // CHECK: %[[E1:.*]] = tcp.get_extent %arg0, 1
  // CHECK: %[[E3:.*]] = tcp.get_extent %arg0, 3
  // CHECK: memref.alloc(%[[E1]], %[[E3]])
  %0 = tcp.alloc_memref %arg0 : !shape.shape -> memref<3x?x5x?x7xf32>
  return
}
