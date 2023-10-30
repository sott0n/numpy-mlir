//RUN: npc-opt -lower-to-npcrt-abi -split-input-file -verify-diagnostics %s | FileCheck %s --dump-input=fail 

// CHECK:      npcrt.module_metadata
// CHECK-NEXT:   npcrt.func_metadata {funcName = @identity, numInputs = 1 : i32, numOutputs = 1 : i32}

// CHECK-LABEL:    func.func @identity(
// CHECK-SAME:                         %[[VAL_0:.*]]: !npcrt.tensor) -> !npcrt.tensor {
// CHECK:            return %[[VAL_0]] : !npcrt.tensor
// CHECK:          }
func.func @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}


// CHECK-LABEL:    func.func @basic(
// CHECK-SAME:                      %[[VAL_0:.*]]: !npcrt.tensor) -> !npcrt.tensor {
// CHECK:             %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_2:.*]] = npcrt.get_extent %[[VAL_0]], %[[VAL_1]]
// CHECK:             %[[VAL_3:.*]] = shape.from_extents %[[VAL_2]]
// CHECK:             %[[VAL_4:.*]] = tcp.alloc_memref %[[VAL_3]] : !shape.shape -> memref<?xf32>
// CHECK:             %[[VAL_5:.*]] = npcrt.to_memref %[[VAL_0]] : memref<*xf32>
// CHECK:             %[[VAL_6:.*]] = memref.cast %[[VAL_5]] : memref<*xf32> to memref<?xf32>
// CHECK:             linalg.copy ins(%[[VAL_6]] : memref<?xf32>) outs(%[[VAL_4]] : memref<?xf32>)
// CHECK:             %[[VAL_7:.*]] = memref.cast %[[VAL_4]] : memref<?xf32> to memref<*xf32>
// CHECK:             %[[VAL_8:.*]] = npcrt.from_memref %[[VAL_7]] : memref<*xf32>
// CHECK:             return %[[VAL_8]] : !npcrt.tensor
// CHECK:           }
func.func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %shape = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %memref = tcp.alloc_memref %shape : !shape.shape -> memref<?xf32>
  memref.tensor_store %arg0, %memref : memref<?xf32>
  %ret = bufferization.to_tensor %memref : memref<?xf32>
  return %ret : tensor<?xf32>
}

// -----

// CHECK: npcrt.global @g dense<7.000000e+00> : tensor<10xf32>
tcp.global @g dense<7.0> : tensor<10xf32>
// CHECK-LABEL: func.func @gets_global
func.func @gets_global() -> tensor<10xf32> {
  // CHECK: %[[GMEMREF:.*]] = npcrt.get_global @g : memref<*xf32>
  // CHECK: %[[ORIGMEMREF:.*]] = memref.cast %[[GMEMREF]] : memref<*xf32> to memref<10xf32>
  // CHECK: %[[RETMEMREF:.*]] = memref.cast %[[ORIGMEMREF]] : memref<10xf32> to memref<*xf32>
  // CHECK: %[[RET:.*]] = npcrt.from_memref %[[RETMEMREF]] : memref<*xf32>
  // CHECK: return %[[RET]] : !npcrt.tensor
  %0 = tcp.get_global_memref @g : memref<10xf32>
  %1 = bufferization.to_tensor %0 : memref<10xf32>
  return %1 : tensor<10xf32>
}

// -----

// expected-error @+1 {{func not expressible with npcrt ABI}}
func.func @unhandled_abi_type_on_public_func(%arg0: i32) {
  return
}
