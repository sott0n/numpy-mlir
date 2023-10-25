// RUN: npc-opt -split-input-file %s | npc-opt | FileCheck --dump-input=fail %s

// CHECK-LABEL: @const_none
func.func @const_none() -> (!basicpy.NoneType) {
  // CHECK: basicpy.singleton : !basicpy.NoneType
  %0 = basicpy.singleton : !basicpy.NoneType
  return %0 : !basicpy.NoneType
}

// -----
// CHECK-LABEL: @const_ellipsis
func.func @const_ellipsis() -> (!basicpy.EllipsisType) {
  // CHECK: basicpy.singleton : !basicpy.EllipsisType
  %0 = basicpy.singleton : !basicpy.EllipsisType
  return %0 : !basicpy.EllipsisType
}
