"""
Tests for GP genetic operators: crossover (cxOnePoint) and mutation (mutUniform).

Validates that:
  - Crossover produces two valid offspring that can be evaluated
  - Mutation produces a valid mutant that can be evaluated
  - Height limits are respected by decorated operators
  - Repeated crossover/mutation doesn't degrade trees
  - Offspring maintain the type-system invariant (root = EnsembleOutput)

Run with debug output:
    pytest unit_tests/test_genetic_ops.py --debug-ops -s
"""
import copy
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

MAX_TREE_HEIGHT = 10


def _setup_creator():
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def _make_toolbox(pset):
    """Build a toolbox with crossover, mutation, and height-limit decorators."""
    _setup_creator()
    tb = base.Toolbox()
    tb.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5,
                type_=gp_types.EnsembleOutput)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("compile", lambda expr: eval(f"lambda: {str(expr)}", pset.context, {}))
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)

    tb.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=MAX_TREE_HEIGHT))
    tb.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=MAX_TREE_HEIGHT))
    return tb


def _compile_lazy(pset, expr):
    code = f"lambda: {str(expr)}"
    return eval(code, pset.context, {})


def _evaluate_tree(pset, individual, X, y, debug_flag=False):
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
    return acc, preds


def _is_valid_tree(ind):
    """Quick structural check: root returns EnsembleOutput, arity sums to 0."""
    if len(ind) == 0:
        return False
    root = ind[0]
    if not (hasattr(root, "ret") and root.ret == gp_types.EnsembleOutput):
        return False
    stack = 1
    for node in ind:
        stack -= 1
        if hasattr(node, "arity"):
            stack += node.arity
    return stack == 0


# ===================================================================
# 1. CROSSOVER
# ===================================================================

class TestCrossover:

    def test_crossover_produces_two_offspring(self, debug):
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        pop = tb.population(n=10)

        dbg_print(debug, "\n=== Crossover: basic offspring test ===")
        successes = 0

        for i in range(0, 10, 2):
            p1 = copy.deepcopy(pop[i])
            p2 = copy.deepcopy(pop[i + 1])
            p1_str = str(p1)
            p2_str = str(p2)

            c1, c2 = tb.mate(p1, p2)

            dbg_print(debug, f"\n  Pair {i // 2}:")
            dbg_print(debug, f"    Parent 1: {p1_str[:80]}...")
            dbg_print(debug, f"    Parent 2: {p2_str[:80]}...")
            dbg_print(debug, f"    Child  1: {str(c1)[:80]}...")
            dbg_print(debug, f"    Child  2: {str(c2)[:80]}...")

            assert len(c1) > 0, "Child 1 is empty"
            assert len(c2) > 0, "Child 2 is empty"
            assert _is_valid_tree(c1), f"Child 1 failed structural check"
            assert _is_valid_tree(c2), f"Child 2 failed structural check"
            successes += 1

        dbg_print(debug, f"\n  {successes}/5 crossover pairs valid ✓")

    def test_crossover_offspring_evaluate(self, debug):
        """Offspring from crossover should produce valid predictions."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        X = _make_random_images()
        y = _make_labels()

        pop = tb.population(n=6)
        dbg_print(debug, "\n=== Crossover: offspring evaluation ===")

        for i in range(0, 6, 2):
            p1 = copy.deepcopy(pop[i])
            p2 = copy.deepcopy(pop[i + 1])
            c1, c2 = tb.mate(p1, p2)

            dbg_print(debug, f"\n  Evaluating crossover pair {i // 2}:")

            for label, child in [("Child1", c1), ("Child2", c2)]:
                if _is_valid_tree(child):
                    acc, preds = _evaluate_tree(pset, child, X, y, debug_flag=debug)
                    assert preds.shape[0] == N_SAMPLES, f"{label}: N mismatch"
                    assert preds.shape[1] == N_CLASSES, f"{label}: class mismatch"
                    dbg_print(debug, f"    {label}: acc={acc:.4f}, height={child.height}")
                else:
                    dbg_print(debug, f"    {label}: INVALID (staticLimit reverted)")
                    # staticLimit may revert to parent – still valid
                    pass

    def test_crossover_height_limit(self, debug):
        """After crossover with decorator, height ≤ MAX_TREE_HEIGHT."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)

        dbg_print(debug, f"\n=== Crossover: height limit (max={MAX_TREE_HEIGHT}) ===")
        pop = tb.population(n=20)

        for _ in range(30):
            p1, p2 = random.sample(pop, 2)
            c1, c2 = tb.mate(copy.deepcopy(p1), copy.deepcopy(p2))
            assert c1.height <= MAX_TREE_HEIGHT, \
                f"Child1 height {c1.height} > {MAX_TREE_HEIGHT}"
            assert c2.height <= MAX_TREE_HEIGHT, \
                f"Child2 height {c2.height} > {MAX_TREE_HEIGHT}"

        dbg_print(debug, "  All 30 crossover trials respected height limit ✓")

    def test_crossover_preserves_root_type(self, debug):
        """Root should always remain EnsembleOutput after crossover."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        pop = tb.population(n=20)

        dbg_print(debug, "\n=== Crossover: root type preservation ===")

        for _ in range(20):
            p1, p2 = random.sample(pop, 2)
            c1, c2 = tb.mate(copy.deepcopy(p1), copy.deepcopy(p2))
            root1 = c1[0]
            root2 = c2[0]
            assert hasattr(root1, "ret") and root1.ret == gp_types.EnsembleOutput
            assert hasattr(root2, "ret") and root2.ret == gp_types.EnsembleOutput

        dbg_print(debug, "  20 crossover trials: root type always EnsembleOutput ✓")


# ===================================================================
# 2. MUTATION
# ===================================================================

class TestMutation:

    def test_mutation_produces_valid_tree(self, debug):
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        pop = tb.population(n=10)

        dbg_print(debug, "\n=== Mutation: structural validity ===")

        for i, ind in enumerate(pop):
            original_str = str(ind)
            mutant, = tb.mutate(copy.deepcopy(ind))

            dbg_print(debug, f"\n  ind {i}:")
            dbg_print(debug, f"    Original: {original_str[:80]}...")
            dbg_print(debug, f"    Mutant:   {str(mutant)[:80]}...")

            assert len(mutant) > 0
            assert _is_valid_tree(mutant), f"Mutant {i} failed structural check"

        dbg_print(debug, "\n  All 10 mutations structurally valid ✓")

    def test_mutation_offspring_evaluate(self, debug):
        """Mutated trees should produce valid predictions."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        X = _make_random_images()
        y = _make_labels()

        pop = tb.population(n=5)
        dbg_print(debug, "\n=== Mutation: offspring evaluation ===")

        for i, ind in enumerate(pop):
            mutant, = tb.mutate(copy.deepcopy(ind))
            dbg_print(debug, f"\n  Evaluating mutant {i}:")

            if _is_valid_tree(mutant):
                acc, preds = _evaluate_tree(pset, mutant, X, y, debug_flag=debug)
                assert preds.shape[0] == N_SAMPLES
                assert preds.shape[1] == N_CLASSES
                dbg_print(debug, f"    acc={acc:.4f}, height={mutant.height}")
            else:
                dbg_print(debug, f"    INVALID (staticLimit reverted)")

    def test_mutation_height_limit(self, debug):
        """After mutation with decorator, height ≤ MAX_TREE_HEIGHT."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        pop = tb.population(n=10)

        dbg_print(debug, f"\n=== Mutation: height limit (max={MAX_TREE_HEIGHT}) ===")

        for _ in range(30):
            ind = random.choice(pop)
            mutant, = tb.mutate(copy.deepcopy(ind))
            assert mutant.height <= MAX_TREE_HEIGHT, \
                f"Mutant height {mutant.height} > {MAX_TREE_HEIGHT}"

        dbg_print(debug, "  30 mutation trials respected height limit ✓")

    def test_mutation_changes_tree(self, debug):
        """Mutation should actually modify the tree (at least sometimes)."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        pop = tb.population(n=10)

        dbg_print(debug, "\n=== Mutation: checking for actual changes ===")
        changed = 0
        for ind in pop:
            original = str(ind)
            mutant, = tb.mutate(copy.deepcopy(ind))
            if str(mutant) != original:
                changed += 1

        dbg_print(debug, f"  {changed}/10 mutations actually changed the tree")
        assert changed > 0, "No mutations changed any tree – likely a bug"

    def test_mutation_preserves_root_type(self, debug):
        """Root should always remain EnsembleOutput after mutation."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        pop = tb.population(n=20)

        dbg_print(debug, "\n=== Mutation: root type preservation ===")

        for ind in pop:
            mutant, = tb.mutate(copy.deepcopy(ind))
            root = mutant[0]
            assert hasattr(root, "ret") and root.ret == gp_types.EnsembleOutput

        dbg_print(debug, "  20 mutation trials: root type always EnsembleOutput ✓")


# ===================================================================
# 3. REPEATED GENETIC OPERATIONS (mini evolution)
# ===================================================================

class TestMiniEvolution:
    """
    Simulate a few generations of evolution to verify that repeated
    crossover + mutation doesn't produce broken trees.
    """

    def test_mini_evolution_loop(self, debug):
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        X = _make_random_images()
        y = _make_labels()
        pop_size = 10
        n_gen = 3
        cx_prob = 0.5
        mut_prob = 0.3

        dbg_print(debug, f"\n=== Mini Evolution: {n_gen} gens, pop={pop_size} ===")

        pop = tb.population(n=pop_size)

        for gen in range(n_gen):
            dbg_print(debug, f"\n  --- Generation {gen} ---")

            # Evaluate fitness
            fitnesses = []
            for ind in pop:
                acc, preds = _evaluate_tree(pset, ind, X, y)
                ind.fitness.values = (acc,)
                fitnesses.append(acc)

            dbg_print(debug, f"    mean fitness = {np.mean(fitnesses):.4f}")
            dbg_print(debug, f"    best fitness = {max(fitnesses):.4f}")

            # Selection
            offspring = tb.select(pop, len(pop))
            offspring = [copy.deepcopy(o) for o in offspring]

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < cx_prob:
                    offspring[i - 1], offspring[i] = tb.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mut_prob:
                    offspring[i], = tb.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Validate all offspring
            for i, ind in enumerate(offspring):
                assert _is_valid_tree(ind), \
                    f"Gen {gen}, ind {i}: invalid tree after genetic ops"
                assert ind.height <= MAX_TREE_HEIGHT, \
                    f"Gen {gen}, ind {i}: height {ind.height} > {MAX_TREE_HEIGHT}"

            pop = offspring

        # Final evaluation
        final_accs = []
        for ind in pop:
            acc, preds = _evaluate_tree(pset, ind, X, y)
            final_accs.append(acc)
            assert preds.shape[0] == N_SAMPLES

        dbg_print(debug, f"\n  Final mean fitness = {np.mean(final_accs):.4f}")
        dbg_print(debug, f"  Final best fitness = {max(final_accs):.4f}")
        dbg_print(debug, "  Mini evolution completed successfully ✓")

    def test_diversity_after_evolution(self, debug):
        """After a few generations, population should have some diversity."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)
        X = _make_random_images()
        y = _make_labels()

        pop = tb.population(n=10)

        dbg_print(debug, "\n=== Diversity check after 2 gens ===")

        for gen in range(2):
            for ind in pop:
                acc, _ = _evaluate_tree(pset, ind, X, y)
                ind.fitness.values = (acc,)

            offspring = tb.select(pop, len(pop))
            offspring = [copy.deepcopy(o) for o in offspring]

            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    offspring[i - 1], offspring[i] = tb.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < 0.3:
                    offspring[i], = tb.mutate(offspring[i])
                    del offspring[i].fitness.values

            pop = offspring

        unique_trees = set(str(ind) for ind in pop)
        dbg_print(debug, f"  Unique trees: {len(unique_trees)}/{len(pop)}")
        # At least some diversity should exist
        assert len(unique_trees) > 1, "All trees are identical after evolution"


# ===================================================================
# 4. EDGE CASES
# ===================================================================

class TestEdgeCases:

    def test_crossover_identical_parents(self, debug):
        """Crossing an individual with itself should still produce valid trees."""
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)

        dbg_print(debug, "\n=== Crossover: identical parents ===")
        ind = tb.individual()
        c1, c2 = tb.mate(copy.deepcopy(ind), copy.deepcopy(ind))
        assert _is_valid_tree(c1)
        assert _is_valid_tree(c2)
        dbg_print(debug, "  Both offspring valid ✓")

    def test_mutation_of_minimal_tree(self, debug):
        """Mutate a tree generated with min depth – should still work."""
        pset = gp_setup.create_primitive_set()
        _setup_creator()

        dbg_print(debug, "\n=== Mutation: minimal tree ===")
        expr = gp_utils.gen_safe(pset, min_=1, max_=2, type_=gp_types.EnsembleOutput)
        ind = creator.Individual(expr)
        dbg_print(debug, f"  Original: height={ind.height}, len={len(ind)}")

        tb = _make_toolbox(pset)
        mutant, = tb.mutate(copy.deepcopy(ind))
        assert _is_valid_tree(mutant)
        dbg_print(debug, f"  Mutant:   height={mutant.height}, len={len(mutant)}")

    def test_crossover_between_modes(self, debug):
        """
        If two trees from the same pset are crossed, result is valid.
        (both basic-mode – we can't cross between different psets)
        """
        pset = gp_setup.create_primitive_set()
        tb = _make_toolbox(pset)

        dbg_print(debug, "\n=== Crossover: different-looking trees ===")
        pop = tb.population(n=20)
        # Pick the shortest and tallest
        pop_sorted = sorted(pop, key=lambda ind: ind.height)
        short = copy.deepcopy(pop_sorted[0])
        tall = copy.deepcopy(pop_sorted[-1])

        dbg_print(debug, f"  Short: height={short.height}, len={len(short)}")
        dbg_print(debug, f"  Tall:  height={tall.height}, len={len(tall)}")

        c1, c2 = tb.mate(short, tall)
        assert _is_valid_tree(c1)
        assert _is_valid_tree(c2)
        dbg_print(debug, f"  Child1: height={c1.height}, Child2: height={c2.height} ✓")
