// RUN: npc-opt < %s -convert-basicpy-to-std | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @binary_expr_add_constant
func.func @binary_expr_add_constant() -> i32 {
    // CHECK: arith.constant
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = basicpy.binary_expr %0 "Add" %1 : (i32, i32) -> i32
    return %2 : i32
}

// -----
// CHECK-LABEL: @binary_expr_add
func.func @binary_expr_add(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: arith.addi
    %0 = basicpy.binary_expr %arg0 "Add" %arg1 : (i32, i32) -> i32
    return %0 : i32
}
