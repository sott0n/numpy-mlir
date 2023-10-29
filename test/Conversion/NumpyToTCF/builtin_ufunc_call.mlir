// RUN: npc-opt <%s -convert-numpy-to-tcf | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @unknownBuiltinUfunc
func.func @unknownBuiltinUfunc(%arg0 : tensor<?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK: numpy.builtin_ufunc_call
  // CHECK-NOT: tcf.add
  %0 = numpy.builtin_ufunc_call<"NON_EXISTING"> (%arg0, %arg1) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----
// CHECK-LABEL: @illegalTernary
func.func @illegalTernary(%arg0 : tensor<?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK: numpy.builtin_ufunc_call
  // CHECK-NOT: tcf.add
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1, %arg0) : (tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----
// CHECK-LABEL: @numpyAdd
func.func @numpyAdd(%arg0 : tensor<?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK: "tcf.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
