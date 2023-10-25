// RUN: npc-opt -split-input-file %s | npc-opt | FileCheck --dump-input=fail %s

// CHECK-LABEL: @binary_compare_eq
func.func @binary_compare_eq() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.binary_compare %0 Eq %1 : i32, i32
    %2 = basicpy.binary_compare %0 "Eq" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// -----
// CHECK-LABEL: @binary_compare_gt
func.func @binary_compare_gt() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.binary_compare %0 Gt %1 : i32, i32
    %2 = basicpy.binary_compare %0 "Gt" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_gte
func.func @binary_compare_gte() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.binary_compare %0 GtE %1 : i32, i32
    %2 = basicpy.binary_compare %0 "GtE" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_in
func.func @binary_compare_in() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_compare %0 In %1 : i32, i32
    %2 = basicpy.binary_compare %0 "In" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_is
func.func @binary_compare_is() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.binary_compare %0 Is %1 : i32, i32
    %2 = basicpy.binary_compare %0 "Is" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_isnot
func.func @binary_compare_isnot() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_compare %0 IsNot %1 : i32, i32
    %2 = basicpy.binary_compare %0 "IsNot" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_lt
func.func @binary_compare_lt() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_compare %0 Lt %1 : i32, i32
    %2 = basicpy.binary_compare %0 "Lt" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_lte
func.func @binary_compare_lte() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_compare %0 LtE %1 : i32, i32
    %2 = basicpy.binary_compare %0 "LtE" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_noteq
func.func @binary_compare_noteq() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_compare %0 NotEq %1 : i32, i32
    %2 = basicpy.binary_compare %0 "NotEq" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_compare_notin
func.func @binary_compare_notin() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.binary_compare %0 NotIn %1 : i32, i32
    %2 = basicpy.binary_compare %0 "NotIn" %1 : i32, i32
    return %2 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_expr_add
func.func @binary_expr_add() -> i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.binary_expr %0 Add %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "Add" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_bitand
func.func @binary_expr_bitand() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 BitAnd %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "BitAnd" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_bitor
func.func @binary_expr_bitor() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 BitOr %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "BitOr" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_bitxor
func.func @binary_expr_bitxor() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 BitXor %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "BitXor" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_div
func.func @binary_expr_div() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 Div %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "Div" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_floordiv
func.func @binary_expr_floordiv() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 FloorDiv %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "FloorDiv" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_lshift
func.func @binary_expr_lshift() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 LShift %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "LShift" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_rshift
func.func @binary_expr_rshift() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 RShift %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "RShift" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_matmult
func.func @binary_expr_matmult() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 Matmult %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "Matmult" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_mod
func.func @binary_expr_mod() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 Mod %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "Mod" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_mult
func.func @binary_expr_mult() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 Mult %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "Mult" %1 : (i32, i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: @binary_expr_sub
func.func @binary_expr_sub() -> i32 {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: basicpy.binary_expr %0 Sub %1 : (i32, i32) -> i32
    %2 = basicpy.binary_expr %0 "Sub" %1 : (i32, i32) -> i32
    return %2 : i32
}

// -----
// CHECK-LABEL: @binary_bool_cast
func.func @binary_bool_cast() -> !basicpy.BoolType {
    %0 = llvm.mlir.constant(true) : i1
    // CHECK: basicpy.bool_cast %0 : i1 -> !basicpy.BoolType
    %1 = basicpy.bool_cast %0 : i1 -> !basicpy.BoolType
    return %1 : !basicpy.BoolType
}

// -----
// CHECK-LABEL: @binary_bool_constant_true
func.func @binary_bool_constant_true() -> !basicpy.BoolType {
    // CHECK: basicpy.bool_constant true
    %0 = basicpy.bool_constant true
    return %0 : !basicpy.BoolType
}

// CHECK-LABEL: @binary_bool_constant_false
func.func @binary_bool_constant_false() -> !basicpy.BoolType {
    // CHECK: basicpy.bool_constant false
    %0 = basicpy.bool_constant false
    return %0 : !basicpy.BoolType
}

// -----
// CHECK-LABEL: @binary_bytes_constant
func.func @binary_bytes_constant() -> !basicpy.BytesType {
    // CHECK: basicpy.bytes_constant "hello"
    %0 = basicpy.bytes_constant "hello"
    return %0 : !basicpy.BytesType
}

// CHECK-LABEL: @binary_bytes_constant_null_pass
func.func @binary_bytes_constant_null_pass() -> !basicpy.BytesType {
    // CHECK: basicpy.bytes_constant ""
    %0 = basicpy.bytes_constant ""
    return %0 : !basicpy.BytesType
}

// -----
// CHECK-LABEL: @exec
func.func @exec() {
    basicpy.exec {
        %0 = llvm.mlir.constant(1 : i64) : i64
        %1 = llvm.mlir.constant(2 : i64) : i64
        %2 = basicpy.binary_expr %0 "Add" %1 : (i64, i64) -> i64
        basicpy.exec_discard %2 : i64
    }
    return
}

// -----
// CHECK-LABEL: @str_constant
func.func @str_constant() -> !basicpy.StrType {
    // CHECK: basicpy.str_constant "foobar"
    %0 = basicpy.str_constant "foobar"
    return %0 : !basicpy.StrType
}

// -----
// CHECK-LABEL: @singleton
func.func @singleton() -> !basicpy.NoneType {
    // CHECK: basicpy.singleton : !basicpy.NoneType
    %0 = basicpy.singleton : !basicpy.NoneType
    return %0 : !basicpy.NoneType
}

// -----
// CHECK-LABEL: @to_boolean
func.func @to_boolean() -> i1 {
    %0 = llvm.mlir.constant(1 : i64) : i64
    // CHECK: basicpy.to_boolean %0 : i64
    %1 = basicpy.to_boolean %0 : i64
    return %1 : i1
}

// -----
// CHECK-LABEL: @unknown_cast
func.func @unknown_cast() -> i64 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: basicpy.unknown_cast %0 : i32 -> i64
    %1 = basicpy.unknown_cast %0 : i32 -> i64
    return %1 : i64
}