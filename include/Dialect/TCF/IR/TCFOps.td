#ifndef TCF_OPS
#define TCF_OPS

include "Dialect/TCF/IR/TCFBase.td"

class TCF_Op<string mnemonic, list<Trait> traits = []> :
        Op<TCF_Dialect, mnemonic, traits>;

// TODO: investigate effects framework for defining error semantics
// TODO: define in a general way across the dialect what "encounters an error" means.

// TODO: verify same dtype?
// TODO: what are the allowable dtypes?
def TCF_AddOp : TCF_Op<"add"> {
  let summary = "Add two tensors.";
  let description = [{
    Add two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def TCF_BatchMatmulOp : TCF_Op<"batch_matmul"> {
  let summary = "Performs a batch of matrix multiplictions.";
  let description = [{
This op, in its simplest case, performs a matrix multiplication between the two operands.
Let the input shapes of the operands have shape:
- `lhs`: `[BLHS..., LHSROWS, LHSCOLS]`
- `rhs`: `[BRHS..., RHSROWS, RHSCOLS]`
Then `result` will have shape `[broadcast(BLHS, BRHS),LHROWS,RHCOLS]`.

This op encounters are error if `LHCOLS` != `RHROWS` or if 
`broadcast(BLHS, BRHS)` is not possible.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

// TODO: represent more general convolutions (via more parameters and also more ops)
// torch.nn.functional has a good summary of frontend needs: https://pytorch.org/docs/stable/nn.functional.html#conv2d
// TODO: describe error conditions
def TCF_Conv2DOp : TCF_Op<"conv_2d"> {
  let summary = "Perform a 2D convolution.";
  let description = [{
This op performs a 2D convolution in the sense typical in deep learning contexts.

The inputs have the following rank structure:
- `input`: `[BATCH, Zin, IN0, IN1]`
- `kernel`: `[Zout, Zin, K0, K1]`
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$kernel
  );
  let results = (outs
    AnyTensor:$result
  );
}

#endif // TCF_OPS
