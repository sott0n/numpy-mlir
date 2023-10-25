#include "InitAll.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include "Dialect/Basicpy/Transforms/Passes.h"
#include "Dialect/Numpy/IR/NumpyDialect.h"
#include "Dialect/Numpy/Transforms/Passes.h"
#include "Dialect/TCF/IR/TCFDialect.h"
//#include "Dialect/TCF/Transforms/Passes.h"
//#include "Dialect/TCP/IR/TCPDialect.h"
//#include "Dialect/Ilrt/IR/IlrtDialect.h"

#include "Conversion/BasicpyToStd/Passes.h"
#include "Conversion/NumpyToTCF/Passes.h"
//#include "Conversion/TCFToTCP/Passes.h"
//#include "Conversion/TCPToLinalg/Passes.h"

//#include "Typing/Transforms/Passes.h"
//#include "E2E/E2E.h"

void mlir::npc::registerAllDialects(DialectRegistry &registry) {
  registry.insert<npc::Basicpy::BasicpyDialect>();
  registry.insert<npc::Numpy::NumpyDialect>();
  registry.insert<npc::tcf::TCFDialect>();
  //registry.insert<il::tcp::TCPDialect>();
  //registry.insert<il::ilrt::IlrtDialect>();
}

namespace mlir {
namespace npc {

namespace Basicpy {
#define GEN_PASS_REGISTRATION
#include "Dialect/Basicpy/Transforms/Passes.h.inc"
} // namespace Basicpy

namespace Numpy {
#define GEN_PASS_REGISTRATION
#include "Dialect/Numpy/Transforms/Passes.h.inc"
} // namespace Numpy

//namespace tcf {
//#define GEN_PASS_REGISTRATION
//#include "Dialect/TCF/Transforms/Passes.h.inc"
//} // namespace tcf
//
namespace Conversion {
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
} // namespace Conversion

//namespace Typing {
//#define GEN_PASS_REGISTRATION
//#include "Typing/Transforms/Passes.h.inc"
//} // namespace Typing
//
//#define GEN_PASS_REGISTRATION
//#include "E2E/Passes.h.inc"

} // namespace npc
} // namespace mlir

void mlir::npc::registerAllPasses() {
  using mlir::Pass;
  //mlir::PassPipelineRegistration<E2ELoweringPipelineOptions>(
  //    "e2e-lowering-pipeline", "E2E lowering pipeline.",
  //    mlir::il::createE2ELoweringPipeline);
  mlir::npc::Basicpy::registerPasses();
  mlir::npc::Numpy::registerPasses();
  //mlir::il::tcf::registerPasses();
  mlir::npc::Conversion::registerPasses();
  //mlir::il::Typing::registerPasses();
  // E2E passes.
  //mlir::il::registerPasses();
}
