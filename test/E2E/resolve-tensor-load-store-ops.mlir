//RUN: npc-opt -resolve-tensor-load-store-ops %s | FileCheck %s --dump-input=fail 

// CHECK-LABEL: func.func @basic
func.func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %shape = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape

  // CHECK: %[[SRCMEMREF:.*]] = tcp.alloc_memref
  %src_memref = tcp.alloc_memref %shape : !shape.shape -> memref<?xf32>
  // CHECK: memref.tensor_store %arg0, %[[SRCMEMREF]]
  memref.tensor_store %arg0, %src_memref : memref<?xf32>
  %src = bufferization.to_tensor %src_memref : memref<?xf32>

  // CHECK: %[[DSTMEMREF:.*]] = tcp.alloc_memref
  %dst_memref = tcp.alloc_memref %shape : !shape.shape -> memref<?xf32>
  // The tensor_store of internally created tensor is eliminated.
  // CHECK-NOT: tensor_store 
  // CHECK: linalg.copy ins(%[[SRCMEMREF]] : memref<?xf32>) outs(%[[DSTMEMREF]] : memref<?xf32>)
  memref.tensor_store %src, %dst_memref : memref<?xf32>
  %ret = bufferization.to_tensor %dst_memref : memref<?xf32>
  
  // The tensor_load feeding into the return remains.
  // CHECK: %[[RET:.*]] = bufferization.to_tensor %[[DSTMEMREF]]
  // CHECK: return %[[RET]]
  return %ret : tensor<?xf32>
}
