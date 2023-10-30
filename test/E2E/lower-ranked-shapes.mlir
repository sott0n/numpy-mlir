// RUN: npc-opt -split-input-file -lower-ranked-shapes %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func.func @broadcast_rank2_rank1
func.func @broadcast_rank2_rank1(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // CHECK-NOT: shape.broadcast
  // CHECK-NOT: tcp.get_extent
  // CHECK-NOT: shape.from_extents
  %0 = shape.from_extents %arg0, %arg1 : index, index
  %1 = shape.from_extents %arg2 : index
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %e0 = tcp.get_extent %2, 0 : !shape.shape
  %e1 = tcp.get_extent %2, 1 : !shape.shape
  return %e0, %e1 : index, index
}

// CHECK-LABEL: func.func @erase_stray_shape_ops
func.func @erase_stray_shape_ops(%arg0: index) {
  // CHECK-NOT: tcp.shape_observe_error
  // CHECK-NOT: shape.from_extents
  %0 = shape.from_extents %arg0 : index
  "tcp.shape_observe_error"(%0) : (!shape.shape) -> none
  return
}

// -----

// CHECK-LABEL: func.func @const_shape
func.func @const_shape() -> index {
  // CHECK-NOT: shape.const_shape
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.const_shape [7] : !shape.shape
  %2 = tcp.get_extent %1, 0 : !shape.shape
  // CHECK: %[[C7:.*]] = arith.constant 7 : index
  // CHECK: return %[[C7]]
  return %2 : index
}
