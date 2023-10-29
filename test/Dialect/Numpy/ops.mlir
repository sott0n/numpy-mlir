// RUN: npc-opt -split-input-file %s | npc-opt | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @create_array_from_tensor
func.func @create_array_from_tensor() -> !numpy.ndarray<[2]:f64> {
  %0 = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %1 = numpy.create_array_from_tensor %0 : (tensor<2xf64>) -> !numpy.ndarray<[2]:f64>
  return %1 : !numpy.ndarray<[2]:f64>
}

// -----
// CHECK-LABEL: @builtin_ufunc
func.func @builtin_ufunc(%arg0 : tensor<3xf64>, %arg1 : tensor<3xf64>) -> tensor<3xf64> {
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

// -----
// CHECK-LABEL: @copy_to_tensor
func.func @copy_to_tensor() -> tensor<*xf64> {
  %0 = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %1 = numpy.create_array_from_tensor %0 : (tensor<2xf64>) -> !numpy.ndarray<*:f64>
  %2 = numpy.copy_to_tensor %1 : (!numpy.ndarray<*:f64>) -> tensor<*xf64>
  return %2 : tensor<*xf64>
}
