#include "PassDetail.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include "Dialect/Basicpy/IR/BasicpyOps.h"
#include "Dialect/Basicpy/Transforms/Passes.h"

#define DEBUG_TYPE "basicpy-type-inference"

using namespace mlir;
using namespace mlir::npc;
using namespace mlir::npc::Basicpy;

namespace {

/// Value type wrapping a type node.
class TypeNode : public llvm::ilist_node<TypeNode> {
public:
  enum class Discrim {
    CONST_TYPE,
    VAR_ORDINAL,
  };

  TypeNode(mlir::Value def, mlir::Type constType)
      : def(def), select(constType), discrim(Discrim::CONST_TYPE) {}
  TypeNode(mlir::Value def, unsigned varOrdinal)
      : def(def), select(varOrdinal), discrim(Discrim::VAR_ORDINAL) {}

  bool operator==(const TypeNode &other) const {
    if (discrim != other.discrim)
      return false;
    switch (discrim) {
    case Discrim::CONST_TYPE:
      return select.constType == other.select.constType;
    case Discrim::VAR_ORDINAL:
      return select.varOrdinal == other.select.varOrdinal;
    }
    return false;
  }

  mlir::Value getDef() const { return def; }
  Discrim getDiscrim() const { return discrim; }

  mlir::Type getConstType() const {
    assert(discrim == Discrim::CONST_TYPE);
    return select.constType;
  }
  unsigned getVarOrdinal() const {
    assert(discrim == Discrim::VAR_ORDINAL);
    return select.varOrdinal;
  }

private:
  mlir::Value def;
  union Select {
    Select(mlir::Type constType) : constType(constType) {}
    Select(unsigned varOrdinal) : varOrdinal(varOrdinal) {}
    mlir::Type constType;
    unsigned varOrdinal;
  } select;
  Discrim discrim;
};

/// A type equation, representing expected equality of types.
/// If the equation is derived from an operation, it is preserved for debugging
/// and messaging.
class TypeEquation : public llvm::ilist_node<class TypeEquation> {
public:
  TypeEquation(TypeNode *left, TypeNode *right, Operation *context)
      : left(left), right(right), context(context) {}

  TypeNode *getLeft() const { return left; }
  TypeNode *getRight() const { return right; }
  Operation *getContext() const { return context; }

private:
  TypeNode *left;
  TypeNode *right;
  Operation *context;
};

raw_ostream &operator<<(raw_ostream &os, const TypeNode &tn) {
  switch (tn.getDiscrim()) {
  case TypeNode::Discrim::CONST_TYPE:
    os << "CONST(" << tn.getConstType() << ")";
    break;
  case TypeNode::Discrim::VAR_ORDINAL:
    os << "VAR(" << tn.getVarOrdinal() << ")";
    break;
  }
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const TypeEquation &eq) {
  os << "[TypeEq left=<" << *eq.getLeft() << ">, right=<" << *eq.getRight()
     << ">: " << *eq.getContext() << "]";
  return os;
}

/// Container for constructing type equations in an HM-like type inference
/// setup.
/// As a first pass, every eligible Value (def) is assigned either a Type or
/// a TypeVar (placefolder).
class TypeEquations {
public:
  TypeEquations() = default;

  // Gets a type node for the given def, creating it if necessary.
  TypeNode *getTypeNode(mlir::Value def) {
    TypeNode *&typeNode = defToNodeMap[def];
    if (typeNode)
      return typeNode;

    if (def.getType().isa<UnknownType>()) {
      // Type variable.
      typeNode = createTypeVar(def);
    } else {
      // Constant type.
      typeNode = createConstType(def, def.getType());
    }
    return typeNode;
  }

  template <typename... Args>
  TypeEquation *addEquation(Args &&...args) {
    TypeEquation *eq = allocator.Allocate<TypeEquation>(1);
    new (eq) TypeEquation(std::forward<Args>(args)...);
    equations.push_back(*eq);
    return eq;
  }

  /// Adds an equality equation for two defs, creating type nodes if necessary.
  void addTypesEqualityEquation(mlir::Value def1, mlir::Value def2,
                                Operation *context) {
    addEquation(getTypeNode(def1), getTypeNode(def2), context);
  }

  /// Print a report of the equations for debugging.
  void report(raw_ostream &os) {
    os << "Type variable map:\n";
    for (auto it : llvm::enumerate(ordinalToVarNode)) {
      os << ":  " << it.index() << " = " << it.value()->getDef() << "\n";
    }
    os << "Type equations:\n";
    os << "---------------\n";
    for (auto &eq : equations) {
      os << "  : " << eq << "\n";
    }
  }

  llvm::simple_ilist<TypeEquation> &getEquations() { return equations; }

  TypeNode *lookupVarOrdinal(unsigned ordinal) {
    assert(ordinal < ordinalToVarNode.size());
    return ordinalToVarNode[ordinal];
  }

private:
  TypeNode *createConstType(mlir::Value def, mlir::Type constType) {
    TypeNode *n = allocator.Allocate<TypeNode>(1);
    new (n) TypeNode(def, constType);
    nodes.push_back(*n);
    return n;
  }
  TypeNode *createTypeVar(mlir::Value def) {
    TypeNode *n = allocator.Allocate<TypeNode>(1);
    new (n) TypeNode(def, nextOrdinal++);
    nodes.push_back(*n);
    ordinalToVarNode.push_back(n);
    assert(ordinalToVarNode.size() == nextOrdinal);
    return n;
  }

  llvm::BumpPtrAllocator allocator;
  llvm::simple_ilist<TypeNode> nodes;
  llvm::simple_ilist<TypeEquation> equations;
  llvm::DenseMap<mlir::Value, TypeNode *> defToNodeMap;
  llvm::SmallVector<TypeNode *, 16> ordinalToVarNode;
  unsigned nextOrdinal = 0;
};

/// (Very) simple type unification. This really isn't advanced enough for
/// anything beyond simple, unambigous programs.
/// It is also terribly inefficient.
class TypeUnifier {
public:
  using SubstMap = llvm::DenseMap<unsigned, TypeNode *>;

  std::optional<SubstMap> unifyEquations(TypeEquations &equations) {
    std::optional<SubstMap> subst;
    subst.emplace();

    for (auto &eq : equations.getEquations()) {
      subst = unify(eq.getLeft(), eq.getRight(), std::move(subst));
      if (!subst) {
        break;
      }
    }
    return subst;
  }

  mlir::Type resolveSubst(TypeNode *typeNode, const std::optional<SubstMap> &subst) {
    if (!subst) {
      return nullptr;
    }
    if (typeNode->getDiscrim() == TypeNode::Discrim::CONST_TYPE) {
      return typeNode->getConstType();
    }
    if (typeNode->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      auto foundIt = subst->find(typeNode->getVarOrdinal());
      if (foundIt != subst->end()) {
        return resolveSubst(foundIt->second, subst);
      } else {
        return nullptr;
      }
    }
    return nullptr;
  }

  std::optional<SubstMap> unify(TypeNode *typeX, TypeNode *typeY,
                           std::optional<SubstMap> subst) {
    LLVM_DEBUG(llvm::dbgs() << "+ UNIFY: " << *typeX << ", " << *typeY << "\n");
    if (!subst) {
      emitError(typeX->getDef().getLoc()) << "cannot unify type";
      emitRemark(typeY->getDef().getLoc()) << "conflicting expression here";
      return std::nullopt;
    } else if (*typeX == *typeY) {
      return subst;
    } else if (typeX->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      return unifyVariable(typeX, typeY, std::move(*subst));
    } else if (typeY->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      return unifyVariable(typeY, typeX, std::move(*subst));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  Unify fallthrough\n");
      return std::nullopt;
    }
  }

  std::optional<SubstMap> unifyVariable(TypeNode *varNode, TypeNode *typeNode,
                                   SubstMap subst) {
    assert(varNode->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL);
    LLVM_DEBUG(llvm::dbgs() << "  - UNIFY VARIABLE: " << *varNode << " <-"
                            << *typeNode << "\n");

    // Var node in subst?
    auto it = subst.find(varNode->getVarOrdinal());
    if (it != subst.end()) {
      TypeNode *found = it->second;
      LLVM_DEBUG(llvm::dbgs() << "   --> FOUND VAR: " << *found << "\n");
      return unify(found, typeNode, std::move(subst));
    }

    // Type node in subst?
    if (typeNode->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      it = subst.find(typeNode->getVarOrdinal());
      if (it != subst.end()) {
        TypeNode *found = it->second;
        LLVM_DEBUG(llvm::dbgs() << "   --> FOUND TYPE: " << *found << "\n");
        return unify(varNode, found, std::move(subst));
      }
    }

    // Does the variable appear in the type?
    if (occursCheck(varNode, typeNode, subst)) {
      LLVM_DEBUG(llvm::dbgs() << "FAILED OCCURS_CHECK\n");
      return std::nullopt;
    }

    // varNode is not yet in subst and cannot simplify typeNode. Extend.
    subst[varNode->getVarOrdinal()] = typeNode;
    return std::move(subst);
  }

  bool occursCheck(TypeNode *varNode, TypeNode *typeNode, SubstMap &subst) {
    if (*varNode == *typeNode)
      return true;

    if (typeNode->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      unsigned typeOrdinal = typeNode->getVarOrdinal();
      auto foundIt = subst.find(typeOrdinal);
      if (foundIt != subst.end()) {
        return occursCheck(varNode, foundIt->second, subst);
      }
    }
    return false;
  }
};

class TypeEquationPopulator {
public:
  TypeEquationPopulator(TypeEquations &equations) : equations(equations) {}

  /// If a return op was visited, this will be one of them.
  Operation *getLastReturnOp() { return funcReturnOp; }

  /// Gets any ReturnLike ops that do no return from the outer function.
  /// This is used to fixup parent SCF ops and the like.
  llvm::SmallVectorImpl<Operation *> &getInnerReturnLikeOps() {
    return innerReturnLikeOps;
  }

  LogicalResult runOnFunction(func::FuncOp funcOp) {
    // Iterate and create type nodes for entry block arguments, as these
    // must be resolved no matter what.
    if (funcOp.getBody().empty())
      return success();

    auto &entryBlock = funcOp.getBody().front();
    for (auto blockArg : entryBlock.getArguments()) {
      equations.getTypeNode(blockArg);
    }

    // Then walk ops, creating equations.
    LLVM_DEBUG(llvm::dbgs() << "POPULATE CHILD OPS:\n");
    auto result = funcOp.walk([&](Operation *childOp) -> WalkResult {
      if (childOp == funcOp)
        return WalkResult::advance();

      LLVM_DEBUG(llvm::dbgs() << "  + POPULATE: " << *childOp << "\n");
      // Special op handling.
      // Many of these (that are not standard ops) should become op
      // interfaces.
      // --------------------
      if (auto op = dyn_cast<arith::SelectOp>(childOp)) {
        // Note that the condition is always i1 and not subject to type
        // inference.
        equations.addTypesEqualityEquation(op.getTrueValue(),
                                           op.getFalseValue(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<ToBooleanOp>(childOp)) {
        // Note that the result is always i1 and not subject to type
        // inference.
        equations.getTypeNode(op.getOperand());
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<scf::IfOp>(childOp)) {
        // Note that the condition is always i1 and not subject to type
        // inference.
        for (auto result : op.getResults()) {
          equations.getTypeNode(result);
        }
        return WalkResult::advance();
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(childOp)) {
        auto scfParentOp = yieldOp->getParentOp();
        if (scfParentOp->getNumResults() != yieldOp.getNumOperands()) {
          yieldOp.emitWarning()
              << "cannot run type inference on yield due to arity mismatch";
          return WalkResult::advance();
        }
        for (auto it :
             llvm::zip(scfParentOp->getResults(), yieldOp.getOperands())) {
          equations.addTypesEqualityEquation(std::get<1>(it), std::get<0>(it),
                                             yieldOp);
        }
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<UnknownCastOp>(childOp)) {
        equations.addTypesEqualityEquation(op.getOperand(), op.getResult(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<BinaryExprOp>(childOp)) {
        // TODO: This should really be applying arithmetic promotion, not
        // strict equality.
        equations.addTypesEqualityEquation(op.getLeft(), op.getRight(), op);
        equations.addTypesEqualityEquation(op.getLeft(), op.getResult(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<BinaryCompareOp>(childOp)) {
        // TODO: This should really be applying arithmetic promotion, not
        // strict equality.
        equations.addTypesEqualityEquation(op.getLeft(), op.getRight(), op);
        return WalkResult::advance();
      }

      // Fallback trait based equations.
      // ---------------------
      // Ensure that constant nodes get assigned a constant type.
      if (childOp->hasTrait<OpTrait::ConstantLike>()) {
        equations.getTypeNode(childOp->getResult(0));
        return WalkResult::advance();
      }
      // Function returns must all have the same types.
      if (childOp->hasTrait<OpTrait::ReturnLike>()) {
        if (childOp->getParentOp() == funcOp) {
          if (funcReturnOp) {
            if (funcReturnOp->getNumOperands() != childOp->getNumOperands()) {
              childOp->emitOpError() << "different arity of function returns";
              return WalkResult::interrupt();
            }
            for (auto it : llvm::zip(funcReturnOp->getOperands(),
                                     childOp->getOperands())) {
              equations.addTypesEqualityEquation(std::get<0>(it),
                                                 std::get<1>(it), childOp);
            }
          }
          funcReturnOp = childOp;
          return WalkResult::advance();
        } else {
          innerReturnLikeOps.push_back(childOp);
        }
      }

      childOp->emitRemark() << "unhandled op in type inference";
      return WalkResult::advance();
    });

    return success(result.wasInterrupted());
  }

private:
  // The last encountered ReturnLike op.
  Operation *funcReturnOp = nullptr;
  llvm::SmallVector<Operation *, 4> innerReturnLikeOps;
  TypeEquations &equations;
};

class FunctionTypeInferencePass
    : public FunctionTypeInferenceBase<FunctionTypeInferencePass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.getBody().empty())
      return;

    TypeEquations equations;
    TypeEquationPopulator p(equations);

    if (!p.runOnFunction(func).failed()) {
      return signalPassFailure();
    }
    LLVM_DEBUG(equations.report(llvm::dbgs()));

    TypeUnifier unifier;
    auto substMap = unifier.unifyEquations(equations);
    if (!substMap) {
      func.emitError() << "type inference failed";
      return signalPassFailure();
    }

    // Apply substitutions.
    LLVM_DEBUG(llvm::dbgs() << "Unification subst:\n");
    LLVM_DEBUG(for (auto it
                    : *substMap) {
      llvm::dbgs() << "  " << it.first << " -> " << *it.second << "\n";
    });
    for (auto it : *substMap) {
      TypeNode *varNode = equations.lookupVarOrdinal(it.first);
      mlir::Type resolvedType = unifier.resolveSubst(it.second, substMap);
      if (!resolvedType) {
        emitError(varNode->getDef().getLoc()) << "unable to infer type";
        continue;
      }
      varNode->getDef().setType(resolvedType);
    }

    // Now rewrite the function type based on actual types of entry block
    // args and the final return op operand.
    auto entryBlockTypes = func.getBody().front().getArgumentTypes();
    SmallVector<mlir::Type, 4> inputTypes(entryBlockTypes.begin(),
                                          entryBlockTypes.end());
    SmallVector<mlir::Type, 4> resultTypes;

    if (p.getLastReturnOp()) {
      auto resultRange = p.getLastReturnOp()->getOperandTypes();
      resultTypes.append(resultRange.begin(), resultRange.end());
    }
    auto funcType = FunctionType::get(&getContext(), inputTypes, resultTypes);
    func.setType(funcType);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::npc::Basicpy::createFunctionTypeInferencePass() {
  return std::make_unique<FunctionTypeInferencePass>();
}
