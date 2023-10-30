// RUN: npc-opt %s -lower-linalg-tensor-to-memref | FileCheck %s --dump-input=fail

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @f
func.func @f(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: %[[LHS:.+]] = tcp.alloc_memref
  // CHECK-DAG: %[[RHS:.+]] = tcp.alloc_memref
  // CHECK-DAG: %[[DST:.+]] = tcp.alloc_memref
  // CHECK: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[LHS]], %[[RHS]] : memref<?xf32>, memref<?xf32>) outs(%[[DST]] : memref<?xf32>) attrs =  {args_in = 2 : i64, args_out = 1 : i64}
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]}
    ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg0: tensor<?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
