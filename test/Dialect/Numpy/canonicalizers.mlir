// RUN: npc-opt -split-input-file %s -canonicalize | FileCheck --dump-input=fail %s

// CHECK-LABEL: @elideCreateRedundantArrayFromTensor
func.func @elideCreateRedundantArrayFromTensor() -> tensor<2xf64> {
  // CHECK: %[[CST:.*]] = arith.constant
  // CHECK-NOT: numpy.create_array_from_tensor
  // CHECK-NOT: numpy.copy_to_tensor
  %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %0 = numpy.create_array_from_tensor %cst : (tensor<2xf64>) -> !numpy.ndarray<[2]:f64>
  %1 = numpy.copy_to_tensor %0 : (!numpy.ndarray<[2]:f64>) -> tensor<2xf64>
  return %1 : tensor<2xf64>
}

// This test verifies that the very trivial elision is not overly aggresive.
// Note that in this example, it is still safe to remove the copy, but the
// analysis has not yet been written to do that safely.
// CHECK-LABEL: @elideCreateRedundantArrayFromTensorNonTrivial
func.func @elideCreateRedundantArrayFromTensorNonTrivial() -> (tensor<2xf64>, tensor<2xf64>) {
  // CHECK: numpy.create_array_from_tensor
  // CHECK: numpy.copy_to_tensor
  %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %0 = numpy.create_array_from_tensor %cst : (tensor<2xf64>) -> !numpy.ndarray<[2]:f64>
  %1 = numpy.copy_to_tensor %0 : (!numpy.ndarray<[2]:f64>) -> tensor<2xf64>
  %2 = numpy.copy_to_tensor %0 : (!numpy.ndarray<[2]:f64>) -> tensor<2xf64>
  return %1, %2 : tensor<2xf64>, tensor<2xf64>
}
