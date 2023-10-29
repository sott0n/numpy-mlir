// RUN: npc-opt -split-input-file %s | npc-opt | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @add
func.func @add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: "tcf.add"
  %0 = "tcf.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}

// -----
// CHECK-LABEL: @batch_matmul
func.func @batch_matmul(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: "tcf.batch_matmul"
  %0 = "tcf.batch_matmul"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}

// -----
// CHECK-LABEL: @conv_2d
func.func @conv_2d(%input: tensor<?x?x?x?xf32>, %kernel: tensor<?x?x?x?xf32>) {
  // CHECK: "tcf.conv_2d"
  %0 = "tcf.conv_2d"(%input, %kernel) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return
}
