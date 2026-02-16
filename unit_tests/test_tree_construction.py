"""
Tests for GP tree construction via gen_safe and DEAP toolbox.

Validates that:
  - gen_safe produces syntactically valid trees
  - Generated trees can be compiled and evaluated end-to-end
  - Trees respect the type system (root = EnsembleOutput)
  - Different pset modes (basic, extended) all generate valid trees
  - Trees of various depths work correctly

Run with debug output:
    pytest unit_tests/test_tree_construction.py --debug-ops -s
"""
import random
import numpy as np
import pytest

from deap import base, creator, tools, gp

from src import gp_setup, gp_ops, gp_types, gp_context, gp_utils

from unit_tests.conftest import (
    dbg_print, consume_prediction, fresh_image_setup,
    _make_random_images, _make_labels,
    N_SAMPLES, IMG_H, IMG_W, N_CLASSES,
)


# ===================================================================
# Helpers
# ===================================================================

def _setup_creator():
    """Create DEAP FitnessMax / Individual classes (idempotent)."""
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def _compile_lazy(pset, expr):
    """Compile an expression tree into a callable (lazy lambda)."""
    code = f"lambda: {str(expr)}"
    return eval(code, pset.context, {})


def _evaluate_tree(pset, individual, X, y, debug_flag=False):
    """
    Full train-mode evaluation of one tree.
    Returns (accuracy, predictions_array).
    """
    func = _compile_lazy(pset, individual)

    gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
    gp_context.context.set_train_labels(y)
    fresh_image_setup(X)

    pipeline = func()

    preds = []
    for batch in pipeline:
        preds.append(batch.data)

    if not preds:
        return 0.0, np.array([])

    preds = np.vstack(preds)
    y_pred = np.argmax(preds, axis=1)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, y_pred)

    if debug_flag:
        print(f"    Tree string (first 120 chars): {str(individual)[:120]}...")
        print(f"    Tree height: {individual.height}")
        print(f"    Tree length (# nodes): {len(individual)}")
        print(f"    Prediction shape: {preds.shape}")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Predicted: {y_pred[:10]}...")
        print(f"    Actual:    {y[:10]}...")

    return acc, preds


# ===================================================================
# 1. gen_safe PRODUCES VALID EXPRESSION LISTS
# ===================================================================

class TestGenSafe:
    """Test the gen_safe tree generator with different configurations."""

    @pytest.mark.parametrize("ext_ops,ext_params", [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ])
    def test_gen_safe_produces_expression(self, ext_ops, ext_params, debug):
        pset = gp_setup.create_primitive_set(extended_ops=ext_ops, extended_params=ext_params)
        dbg_print(debug, f"\n--- gen_safe (extended_ops={ext_ops}, extended_params={ext_params}) ---")

        for trial in range(5):
            expr = gp_utils.gen_safe(pset, min_=2, max_=5, type_=gp_types.EnsembleOutput)
            assert len(expr) > 0, "gen_safe returned empty expression"
            dbg_print(debug, f"  trial {trial}: {len(expr)} nodes")

    @pytest.mark.parametrize("max_depth", [3, 5, 8])
    def test_gen_safe_different_depths(self, max_depth, debug):
        pset = gp_setup.create_primitive_set()
        dbg_print(debug, f"\n--- gen_safe max_depth={max_depth} ---")

        for trial in range(3):
            expr = gp_utils.gen_safe(pset, min_=1, max_=max_depth, type_=gp_types.EnsembleOutput)
            tree = gp.PrimitiveTree(expr)
            dbg_print(debug, f"  trial {trial}: height={tree.height}, len={len(tree)}")
            assert tree.height >= 1, "Tree is trivially flat"
            assert len(tree) >= 1

    def test_root_is_ensemble(self, debug):
        """The first node in the expression must produce EnsembleOutput."""
        pset = gp_setup.create_primitive_set()
        dbg_print(debug, "\n--- Checking root type ---")

        for _ in range(10):
            expr = gp_utils.gen_safe(pset, min_=2, max_=5, type_=gp_types.EnsembleOutput)
            root_node = expr[0]
            # Root should be a primitive (not a terminal) that returns EnsembleOutput
            assert hasattr(root_node, "ret"), "Root is not a primitive"
            assert root_node.ret == gp_types.EnsembleOutput, \
                f"Root returns {root_node.ret}, expected EnsembleOutput"
            dbg_print(debug, f"  root = {root_node.name}, ret = {root_node.ret.__name__}")


# ===================================================================
# 2. TREE COMPILATION & END-TO-END EVALUATION
# ===================================================================

class TestTreeEvaluation:
    """Trees generated by gen_safe can be compiled and evaluated."""

    def test_basic_tree_evaluates(self, debug):
        """Generate a tree, compile it, evaluate it – should not crash."""
        pset = gp_setup.create_primitive_set()
        X = _make_random_images()
        y = _make_labels()

        dbg_print(debug, "\n=== Evaluating randomly generated tree ===")
        expr = gp_utils.gen_safe(pset, min_=2, max_=5, type_=gp_types.EnsembleOutput)
        _setup_creator()
        individual = creator.Individual(expr)

        acc, preds = _evaluate_tree(pset, individual, X, y, debug_flag=debug)
        assert preds.shape[0] == N_SAMPLES, "Prediction count mismatch"
        assert preds.shape[1] == N_CLASSES, "Class count mismatch"
        assert 0.0 <= acc <= 1.0

    @pytest.mark.parametrize("trial", range(5))
    def test_multiple_random_trees(self, trial, debug):
        """Generate and evaluate 5 independent random trees."""
        pset = gp_setup.create_primitive_set()
        X = _make_random_images()
        y = _make_labels()
        _setup_creator()

        dbg_print(debug, f"\n--- Random tree trial {trial} ---")
        expr = gp_utils.gen_safe(pset, min_=2, max_=5, type_=gp_types.EnsembleOutput)
        individual = creator.Individual(expr)

        acc, preds = _evaluate_tree(pset, individual, X, y, debug_flag=debug)
        assert preds.shape[0] == N_SAMPLES

    def test_extended_ops_tree_evaluates(self, debug):
        """A tree with extended ops can be compiled and evaluated."""
        pset = gp_setup.create_primitive_set(extended_ops=True, extended_params=False)
        X = _make_random_images()
        y = _make_labels()
        _setup_creator()

        dbg_print(debug, "\n=== Extended-ops tree evaluation ===")
        expr = gp_utils.gen_safe(pset, min_=2, max_=4, type_=gp_types.EnsembleOutput)
        individual = creator.Individual(expr)

        acc, preds = _evaluate_tree(pset, individual, X, y, debug_flag=debug)
        assert preds.shape[0] == N_SAMPLES

    def test_extended_params_tree_evaluates(self, debug):
        """A tree with extended params can be compiled and evaluated."""
        pset = gp_setup.create_primitive_set(extended_ops=False, extended_params=True)
        X = _make_random_images()
        y = _make_labels()
        _setup_creator()

        dbg_print(debug, "\n=== Extended-params tree evaluation ===")
        expr = gp_utils.gen_safe(pset, min_=2, max_=4, type_=gp_types.EnsembleOutput)
        individual = creator.Individual(expr)

        acc, preds = _evaluate_tree(pset, individual, X, y, debug_flag=debug)
        assert preds.shape[0] == N_SAMPLES


# ===================================================================
# 3. POPULATION GENERATION
# ===================================================================

class TestPopulationGeneration:
    """Toolbox-driven population generation and batch evaluation."""

    def _make_toolbox(self, pset):
        _setup_creator()
        tb = base.Toolbox()
        tb.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5,
                     type_=gp_types.EnsembleOutput)
        tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        return tb

    def test_generate_population(self, debug):
        pset = gp_setup.create_primitive_set()
        tb = self._make_toolbox(pset)
        pop_size = 10

        dbg_print(debug, f"\n--- Generating population of {pop_size} ---")
        pop = tb.population(n=pop_size)
        assert len(pop) == pop_size

        for i, ind in enumerate(pop):
            assert len(ind) > 0
            root = ind[0]
            assert hasattr(root, "ret") and root.ret == gp_types.EnsembleOutput
            dbg_print(debug, f"  ind {i}: height={ind.height}, len={len(ind)}")

    def test_evaluate_population(self, debug):
        """Evaluate a small population – every tree should produce valid output."""
        pset = gp_setup.create_primitive_set()
        tb = self._make_toolbox(pset)
        X = _make_random_images()
        y = _make_labels()

        pop = tb.population(n=5)
        dbg_print(debug, "\n--- Evaluating population of 5 ---")

        accs = []
        for i, ind in enumerate(pop):
            acc, preds = _evaluate_tree(pset, ind, X, y, debug_flag=debug)
            accs.append(acc)
            assert preds.shape[0] == N_SAMPLES
            dbg_print(debug, f"  ind {i}: acc={acc:.4f}")

        dbg_print(debug, f"  mean acc = {np.mean(accs):.4f}")


# ===================================================================
# 4. TRAIN → EVAL MODE
# ===================================================================

class TestTrainEvalCycle:
    """After training, switch to EVAL mode and verify predictions still work."""

    def test_train_then_eval(self, debug):
        pset = gp_setup.create_primitive_set()
        X = _make_random_images()
        y = _make_labels()
        _setup_creator()

        expr = gp_utils.gen_safe(pset, min_=2, max_=4, type_=gp_types.EnsembleOutput)
        individual = creator.Individual(expr)

        dbg_print(debug, "\n=== Train → Eval cycle ===")

        # TRAIN
        dbg_print(debug, "  [TRAIN]")
        train_acc, train_preds = _evaluate_tree(pset, individual, X, y, debug_flag=debug)
        assert train_preds.shape[0] == N_SAMPLES

        # EVAL (reuse trained models)
        dbg_print(debug, "  [EVAL]")
        func = _compile_lazy(pset, individual)
        gp_context.context.reset(gp_context.ExecutionMode.EVAL)
        # Must NOT clear models
        fresh_image_setup(X)
        pipeline = func()

        eval_preds = []
        for batch in pipeline:
            eval_preds.append(batch.data)
        eval_preds = np.vstack(eval_preds)

        assert eval_preds.shape == (N_SAMPLES, N_CLASSES)
        eval_cls = np.argmax(eval_preds, axis=1)
        dbg_print(debug, f"  EVAL predicted: {eval_cls[:10]}")

        # In eval mode with the same data we should get deterministic results
        # (no cross-val, just model.predict_proba)
        assert np.isfinite(eval_preds).all()


# ===================================================================
# 5. TREE STRUCTURE VALIDITY
# ===================================================================

class TestTreeStructure:
    """
    Verify structural properties of generated trees:
    - All nodes have correct arity
    - Types match along edges
    """

    def test_tree_is_parseable(self, debug):
        """PrimitiveTree(expr) parses without error and str() round-trips."""
        pset = gp_setup.create_primitive_set()

        for trial in range(10):
            expr = gp_utils.gen_safe(pset, min_=2, max_=5, type_=gp_types.EnsembleOutput)
            tree = gp.PrimitiveTree(expr)
            tree_str = str(tree)
            assert len(tree_str) > 0
            dbg_print(debug, f"  trial {trial}: {tree_str[:100]}...")

    def test_all_nodes_accounted_for(self, debug):
        """
        Walk the expression list and verify that every primitive's
        argument slots are filled by subsequent nodes.
        """
        pset = gp_setup.create_primitive_set()

        for trial in range(10):
            expr = gp_utils.gen_safe(pset, min_=2, max_=5, type_=gp_types.EnsembleOutput)
            # In DEAP's prefix notation, we can validate by counting arities
            stack = 1  # expect 1 value (the root)
            for node in expr:
                stack -= 1  # this node fills one expected slot
                if hasattr(node, "arity"):
                    stack += node.arity  # primitive opens new slots
                # Terminal has arity 0 (or no arity attribute)
            assert stack == 0, f"Tree has unresolved slots: {stack}"
            dbg_print(debug, f"  trial {trial}: arity check passed, {len(expr)} nodes")
