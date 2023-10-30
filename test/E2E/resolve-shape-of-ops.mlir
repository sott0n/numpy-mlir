//RUN: npc-opt -resolve-shape-of-ops -split-input-file -verify-diagnostics %s | FileCheck %s --dump-input=fail 

// CHECK-LABEL: func.func @basic
func.func @basic(%arg0: !shape.shape) -> !shape.shape {
  %memref = tcp.alloc_memref %arg0 : !shape.shape -> memref<?xf32>
  %tensor = bufferization.to_tensor %memref : memref<?xf32>
  %shape = "shape.shape_of"(%tensor) : (tensor<?xf32>) -> !shape.shape
  // CHECK: return %arg0
  return %shape : !shape.shape
}

// -----

// CHECK-LABEL: func.func @arg_unresolved_ok
func.func @arg_unresolved_ok(%arg0: tensor<?xf32>) -> !shape.shape {
  %0 = "shape.shape_of"(%arg0): (tensor<?xf32>) -> !shape.shape
  return %0 : !shape.shape
}

// -----

func.func @bb_arg_unresolved_not_ok(%arg0: i1, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> !shape.shape {
  cf.cond_br %arg0, ^bb1(%arg1: tensor<?xf32>), ^bb1(%arg2: tensor<?xf32>)
^bb1(%bbarg: tensor<?xf32>):
  // expected-error @+1 {{failed to legalize operation 'shape.shape_of' that was explicitly marked illegal}}
  %0 = "shape.shape_of"(%bbarg): (tensor<?xf32>) -> !shape.shape
  return %0 : !shape.shape
}
