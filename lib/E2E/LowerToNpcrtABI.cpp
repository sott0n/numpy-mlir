#include "PassDetail.h"
#include "E2E/E2E.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "Dialect/Npcrt/IR/NpcrtDialect.h"
#include "Dialect/Npcrt/IR/NpcrtOps.h"
#include "Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::npc;

//===----------------------------------------------------------------------===//
// Creating module metadata.
//===----------------------------------------------------------------------===//

// Returns true if the function signature can be expressed with the ilrt ABI.
static bool expressibleWithIlrtABI(FunctionType type) {
  // Currently, only tensor types can be expressed at ilrt ABI boundaries.
  return llvm::all_of(
      llvm::concat<const Type>(type.getInputs(), type.getResults()),
      [](Type t) { return t.isa<TensorType>(); });
}

static LogicalResult createModuleMetadata(ModuleOp module) {
  auto moduleMetadata = OpBuilder::atBlockBegin(module.getBody())
                            .create<npcrt::ModuleMetadataOp>(module.getLoc());
  moduleMetadata.getMetadatas().push_back(new Block);
  Block &metadatas = moduleMetadata.getMetadatas().front();
  OpBuilder::atBlockEnd(&metadatas)
      .create<npcrt::ModuleMetadataTerminatorOp>(module.getLoc());

  SymbolTable symbolTable(module);
  auto builder = OpBuilder::atBlockBegin(&metadatas);
  for (auto func : module.getOps<func::FuncOp>()) {
    if (symbolTable.getSymbolVisibility(func) !=
        SymbolTable::Visibility::Public) {
      continue;
    }
    builder.create<npcrt::FuncMetadataOp>(
        func.getLoc(), SymbolRefAttr::get(func),
        builder.getI32IntegerAttr(func.getNumArguments()),
        builder.getI32IntegerAttr(func.getNumResults()));

    if (!expressibleWithIlrtABI(func.getFunctionType()))
      return func.emitError() << "func not expressible with npcrt ABI";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect conversion.
//===----------------------------------------------------------------------===//

namespace {
class LowerTensorStoreOp : public OpConversionPattern<memref::TensorStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::TensorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getMemref().getType().cast<MemRefType>();
    Value abiMemref = rewriter.create<npcrt::ToMemrefOp>(
        op.getLoc(),
        UnrankedMemRefType::get(memrefType.getElementType(), /*memorySpace=*/0),
        adaptor.getTensor());
    auto memref =
        rewriter.create<memref::CastOp>(op.getLoc(), memrefType, abiMemref);
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, memref.getDest(),
                                                adaptor.getMemref());
    return success();
  }
};
} // namespace

namespace {
class LowerTensorLoadOp
    : public OpConversionPattern<bufferization::ToTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto abiMemref = rewriter.create<memref::CastOp>(
        op.getLoc(),
        UnrankedMemRefType::get(
            adaptor.getMemref().getType().cast<MemRefType>().getElementType(),
            /*memorySpace=*/0),
        adaptor.getMemref());
    rewriter.replaceOpWithNewOp<npcrt::FromMemrefOp>(
        op, rewriter.getType<npcrt::TensorType>(), abiMemref);
    return success();
  }
};
} // namespace

namespace {
class LowerShapeOfOp : public OpConversionPattern<shape::ShapeOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::ShapeOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: For now ilrt only supports ranked tensor types for its shape
    // lowering, since we don't have a runtime shape struct and lower all shapes
    // to individual SSA values.
    auto tensorType = op.getArg().getType().cast<RankedTensorType>();
    SmallVector<Value, 6> extents;
    for (int i = 0, e = tensorType.getRank(); i < e; i++) {
      auto ci = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getI32IntegerAttr(i));
      extents.push_back(rewriter.create<npcrt::GetExtentOp>(
          op.getLoc(), rewriter.getIndexType(), adaptor.getArg(), ci));
    }
    rewriter.replaceOpWithNewOp<shape::FromExtentsOp>(
        op, rewriter.getType<shape::ShapeType>(), extents);
    return success();
  }
};
} // namespace

namespace {
class ApplyShapeBroadcastOp : public OpConversionPattern<shape::BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Modify type the ExtentTensor to !shape.shape to apply modified previous
    // operands by LowerShapeOfOp.
    rewriter.replaceOpWithNewOp<shape::BroadcastOp>(
        op, rewriter.getType<shape::ShapeType>(), adaptor.getOperands(),
        nullptr);
    return success();
  }
};
} // namespace

namespace {
class ApplyShapeObserveErrorOp
    : public OpConversionPattern<tcp::ShapeObserveErrorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::ShapeObserveErrorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tcp::ShapeObserveErrorOp>(
        op, adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
class ApplyTcpAllocMemRef : public OpConversionPattern<tcp::AllocMemRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::AllocMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tcp::AllocMemRefOp>(
        op, op.getMemref().getType(), adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
class ApplyTcpGetExtent : public OpConversionPattern<tcp::GetExtentOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GetExtentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tcp::GetExtentOp>(op, adaptor.getShape(),
                                                  adaptor.getDim());
    return success();
  }
};
} // namespace

namespace {
class LowerGlobalOp : public OpConversionPattern<tcp::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<npcrt::GlobalOp>(op, op.getName(),
                                                op.getValue());
    return success();
  }
};
} // namespace

namespace {
class LowerGetGlobalMemrefOp
    : public OpConversionPattern<tcp::GetGlobalMemrefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GetGlobalMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto abiMemrefType = UnrankedMemRefType::get(
        op.getType().cast<ShapedType>().getElementType(), /*memorySpace=*/0);
    auto abiMemref = rewriter.create<npcrt::GetGlobalOp>(
        op.getLoc(), abiMemrefType, op.getGlobal());
    // Cast back to the original type.
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), abiMemref);
    return success();
  }
};
} // namespace

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  TensorTypeConverter() {
    addConversion(convertType);
    addConversion([](npcrt::TensorType type) { return type; });
  }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    if (auto type = npcrt::TensorType::get(t.getContext())) {
      results.push_back(type);
      return success();
    }

    results.push_back(t);
    return success();
  }

  /// Return true if the inputs and outputs of the given function type are
  /// legal. Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.
  bool isSignatureLegal(FunctionType funcType) {
    return llvm::all_of(
        llvm::concat<const Type>(funcType.getInputs(), funcType.getResults()),
        [this](Type type) { return isLegal(type); });
  }
};

static LogicalResult doDialectConversion(ModuleOp module) {
  auto *context = module.getContext();

  TensorTypeConverter converter;

  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return converter.isLegal(op.getOperandTypes());
  });

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateReturnOpTypeConversionPattern(patterns, converter);

  patterns.insert<LowerTensorStoreOp>(context);
  target.addIllegalOp<memref::TensorStoreOp>();
  target.addLegalOp<npcrt::ToMemrefOp>();
  target.addLegalOp<linalg::CopyOp>();
  target.addLegalOp<memref::CastOp>();

  patterns.insert<LowerTensorLoadOp>(context);
  target.addIllegalOp<bufferization::ToTensorOp>();
  target.addLegalOp<npcrt::FromMemrefOp>();

  patterns.insert<LowerShapeOfOp>(context);
  patterns.insert<ApplyShapeBroadcastOp>(context);
  patterns.insert<ApplyShapeObserveErrorOp>(context);
  patterns.insert<ApplyTcpAllocMemRef>(context);
  patterns.insert<ApplyTcpGetExtent>(context);
  target.addIllegalOp<shape::FromExtentTensorOp>();
  target.addIllegalOp<shape::ShapeOfOp>();
  target.addDynamicallyLegalOp<shape::BroadcastOp>(
      [](shape::BroadcastOp op) -> bool {
        return !shape::isExtentTensorType(op->getOperand(0).getType());
      });
  target.addDynamicallyLegalOp<tcp::ShapeObserveErrorOp>(
      [](tcp::ShapeObserveErrorOp op) -> bool {
        auto prevOperand = op->getOperand(0);
        if (!shape::isExtentTensorType(prevOperand.getType()))
          return true;

        if (mlir::isa<shape::BroadcastOp>(prevOperand.getDefiningOp()))
          return false;

        return true;
      });
  target.addDynamicallyLegalOp<tcp::AllocMemRefOp>(
      [](tcp::AllocMemRefOp op) -> bool {
        auto prevOperand = op->getOperand(0);
        if (!shape::isExtentTensorType(prevOperand.getType()))
          return true;

        auto prevDefiningOp = prevOperand.getDefiningOp();
        if (mlir::isa<shape::BroadcastOp>(prevDefiningOp))
          return false;
        if (mlir::isa<shape::ShapeOfOp>(prevDefiningOp))
          return false;

        return true;
      });
  target.addDynamicallyLegalOp<tcp::GetExtentOp>(
      [](tcp::GetExtentOp op) -> bool {
        auto prevOperand = op->getOperand(0);
        if (!shape::isExtentTensorType(prevOperand.getType()))
          return true;

        if (mlir::isa<shape::ConstShapeOp>(prevOperand.getDefiningOp()))
          return true;

        return false;
      });
  target.addLegalOp<arith::ConstantOp>();
  target.addLegalOp<npcrt::GetExtentOp>();
  target.addLegalOp<shape::FromExtentsOp>();

  patterns.insert<LowerGlobalOp>(context);
  target.addIllegalOp<tcp::GlobalOp>();
  target.addLegalOp<npcrt::GlobalOp>();

  patterns.insert<LowerGetGlobalMemrefOp>(context);
  target.addIllegalOp<tcp::GetGlobalMemrefOp>();
  target.addLegalOp<npcrt::GetGlobalOp>();

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace {
// This pass lowers the public ABI of the module to the primitives exposed by
// the npcrt dialect.
class LowerToNpcrtABI : public LowerToNpcrtABIBase<LowerToNpcrtABI> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Before we lower anything, capture any needed metadata about the argument
    // lists that will be needed for safely invoking the raw runtime functions
    // later. (for example, number of expected arguments/results, types, etc.)
    if (failed(createModuleMetadata(module)))
      return signalPassFailure();

    // Now do the actual conversion / lowering.
    if (failed(doDialectConversion(module)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::npc::createLowerToNpcrtABIPass() {
  return std::make_unique<LowerToNpcrtABI>();
}
