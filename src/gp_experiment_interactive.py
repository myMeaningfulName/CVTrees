"""
Interactive Parallel Experiment Runner for GP Evolution.

This module provides an interactive version of the parallel experiment runner 
where the user manually controls the population via JSON files instead of using
automatic GP operations (crossover/mutation).

Key features:
- User-controlled population: No automatic crossover/mutation
- JSON-based population input: S-expressions loaded from JSON files
- Interactive workflow: Script waits for user approval between generations
- Robust parsing: Helpful error messages for syntax issues
- Graceful exit: User can quit anytime and get progress chart
- SLURM-aware: Same parallel evaluation as gp_experiment_parallel

Workflow:
1. Script creates gen_0/population.json with empty template
2. User fills in S-expressions and signals "ready"
3. Script parses, validates, and evaluates population in parallel
4. Results saved to gen_0/, script creates gen_1/population.json
5. User chooses: continue (fill next population) or quit (generate summary)
"""

import os
import json
import time
import numpy as np
from deap import base, creator, tools, gp
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

from src import gp_setup, gp_ops, gp_types, gp_context, gp_draw
from src.gp_parallel import (
    ParallelEvaluator, 
    EvaluationResult, 
    get_available_workers
)
from src.gp_logging import (
    setup_experiment_logging, get_logger, log_memory,
    start_heartbeat, stop_heartbeat, log_exception,
    TimingContext, get_system_memory_info,
)


class SExpressionParser:
    """
    Parser for S-expressions (DEAP GP tree string representation).
    
    Converts strings like "rf_classification(hog_features(GetRed), 100, 50)"
    into DEAP PrimitiveTree individuals.
    
    Note: DEAP's built-in from_string() doesn't handle typed primitives well
    when numeric literals are used (it expects ephemeral constant types like Trees, Depth).
    This parser handles numeric literals by inferring the expected type from context.
    """
    
    def __init__(self, pset: gp.PrimitiveSetTyped):
        """
        Initialize the parser with a primitive set.
        
        Args:
            pset: DEAP PrimitiveSetTyped that defines valid primitives/terminals
        """
        self.pset = pset
        
        # Build lookup tables for primitives and terminals by name
        self._primitives = {}
        self._terminals = {}
        
        for ret_type, prims in pset.primitives.items():
            for prim in prims:
                self._primitives[prim.name] = prim
        
        for ret_type, terms in pset.terminals.items():
            for term in terms:
                if hasattr(term, 'name'):
                    self._terminals[term.name] = term
    
    def parse(self, expr_str: str) -> Tuple[Optional[gp.PrimitiveTree], Optional[str]]:
        """
        Parse an S-expression string into a DEAP PrimitiveTree.
        
        Args:
            expr_str: String representation of the tree (S-expression)
            
        Returns:
            Tuple of (tree, error_message)
            - On success: (PrimitiveTree, None)
            - On failure: (None, error_description)
        """
        expr_str = expr_str.strip()
        
        if not expr_str:
            return None, "Empty expression"
        
        try:
            tokens = self._tokenize(expr_str)
            tree_list, remaining, _ = self._parse_tokens(tokens, expected_type=None)
            
            if remaining:
                return None, f"Unexpected tokens after expression: {' '.join(str(t) for t in remaining[:5])}..."
            
            # Create the DEAP individual
            if hasattr(creator, 'Individual'):
                individual = creator.Individual(tree_list)
            else:
                individual = gp.PrimitiveTree(tree_list)
            
            return individual, None
            
        except Exception as e:
            return None, str(e)
    
    def _tokenize(self, expr_str: str) -> List[str]:
        """Tokenize an S-expression string into a list of tokens."""
        tokens = []
        i = 0
        n = len(expr_str)
        
        while i < n:
            c = expr_str[i]
            
            # Skip whitespace and commas
            if c.isspace() or c == ',':
                i += 1
                continue
            
            # Parentheses
            if c in '()':
                tokens.append(c)
                i += 1
                continue
            
            # Numbers (int, float, negative, scientific)
            if c.isdigit() or (c == '-' and i + 1 < n and (expr_str[i+1].isdigit() or expr_str[i+1] == '.')):
                j = i + (1 if c == '-' else 0)
                while j < n and (expr_str[j].isdigit() or expr_str[j] == '.'):
                    j += 1
                if j < n and expr_str[j] in 'eE':
                    j += 1
                    if j < n and expr_str[j] in '+-':
                        j += 1
                    while j < n and expr_str[j].isdigit():
                        j += 1
                tokens.append(expr_str[i:j])
                i = j
                continue
            
            # Identifiers
            if c.isalpha() or c == '_':
                j = i
                while j < n and (expr_str[j].isalnum() or expr_str[j] == '_'):
                    j += 1
                tokens.append(expr_str[i:j])
                i = j
                continue
            
            raise ValueError(f"Unexpected character '{c}' at position {i}")
        
        return tokens
    
    def _parse_tokens(self, tokens: List[str], expected_type) -> Tuple[List, List[str], Any]:
        """
        Parse tokens into a DEAP-compatible tree list.
        
        Args:
            tokens: List of tokens to parse
            expected_type: The expected return type (used for numeric literal inference)
        
        Returns:
            Tuple of (tree_nodes_list, remaining_tokens, return_type)
        """
        if not tokens:
            raise ValueError("Unexpected end of expression")
        
        token = tokens[0]
        remaining = tokens[1:]
        
        # Check if it's a primitive (function call)
        if token in self._primitives:
            prim = self._primitives[token]
            tree = [prim]
            
            if not remaining or remaining[0] != '(':
                raise ValueError(f"Expected '(' after primitive '{token}'")
            remaining = remaining[1:]
            
            # Parse arguments with expected types
            for arg_idx, arg_type in enumerate(prim.args):
                if not remaining:
                    raise ValueError(f"'{token}' expects {len(prim.args)} arguments, got {arg_idx}")
                
                arg_tree, remaining, _ = self._parse_tokens(remaining, expected_type=arg_type)
                tree.extend(arg_tree)
            
            if not remaining or remaining[0] != ')':
                raise ValueError(f"Expected ')' after arguments of '{token}'")
            remaining = remaining[1:]
            
            return tree, remaining, prim.ret
        
        # Check if it's a terminal
        if token in self._terminals:
            term = self._terminals[token]
            return [term], remaining, getattr(term, 'ret', None)
        
        # Check if it's a numeric constant
        try:
            if '.' in token or 'e' in token.lower():
                value = float(token)
            else:
                value = int(token)
            
            # Create a terminal with the expected type
            # The expected_type tells us what kind of ephemeral this should be
            term = gp.Terminal(value, symbolic=False, ret=expected_type or type(value))
            return [term], remaining, expected_type or type(value)
            
        except ValueError:
            pass
        
        raise ValueError(f"Unknown primitive or terminal: '{token}'")
    
    def validate_structure(self, individual: gp.PrimitiveTree) -> Tuple[bool, Optional[str]]:
        """
        Validate that the tree structure is correct.
        
        Args:
            individual: DEAP PrimitiveTree to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to compile the tree
            func = gp.compile(individual, self.pset)
            return True, None
        except Exception as e:
            return False, f"Compilation error: {str(e)}"


class InteractiveExperimentRunner:
    """
    Interactive experiment runner for GP evolution with user-controlled population.
    
    Instead of automatic GP operations, the user provides population members
    via JSON files. The script waits for user input between generations.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "experiments",
                 n_workers: int = None):
        """
        Initialize the interactive experiment runner.
        
        Args:
            experiment_name: Name of the experiment (used for output directory)
            output_dir: Base output directory
            n_workers: Number of worker processes (default: auto-detect)
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.base_dir = os.path.join(output_dir, experiment_name)
        self.n_workers = n_workers
        
        if os.path.exists(self.base_dir):
            print(f"Warning: Experiment directory {self.base_dir} exists.")
        else:
            os.makedirs(self.base_dir)
        
        # Setup logging & heartbeat
        log_file = setup_experiment_logging(experiment_name, output_dir)
        self.logger = get_logger("interactive")
        self.logger.info("InteractiveExperimentRunner created  |  experiment=%s", experiment_name)
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.batch_size = 32
        self.pset = None
        self.parser = None
        
        # Global stats tracking
        self.all_generations_stats = []
        self.best_individual_overall = None
        self.best_fitness_overall = -1.0
        self.current_generation = 0
        
        # Parallel evaluator (initialized in run())
        self.evaluator = None
        
    def setup_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """Setup training, validation, and test data."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.logger.info("Data loaded  |  train=%s  val=%s  test=%s  |  batch_size=%d",
                         X_train.shape, X_val.shape, X_test.shape, batch_size)
        log_memory(label="after-data-load", logger=self.logger)
        
    def setup_gp(self, pset: gp.PrimitiveSetTyped):
        """
        Setup GP primitive set and parser.
        
        Args:
            pset: DEAP primitive set defining the GP language
        """
        self.pset = pset
        self.parser = SExpressionParser(pset)
        
        # Setup DEAP creator
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    def _create_population_template(self, gen_num: int, previous_results: List[Dict] = None) -> str:
        """
        Create a population.json template file for the given generation.
        
        Args:
            gen_num: Generation number
            previous_results: Optional list of results from previous generation
            
        Returns:
            Path to the created JSON file
        """
        gen_dir = os.path.join(self.base_dir, f"gen_{gen_num}")
        os.makedirs(gen_dir, exist_ok=True)
        
        json_path = os.path.join(gen_dir, "population.json")
        
        # Create template with helpful comments in description
        template = {
            "_description": (
                "Fill in the 'individuals' array with S-expression strings. "
                "Each S-expression should be a valid GP tree that returns a Prediction type. "
                "Example: rf_classification(hog_features(GetRed), 100, 50)"
            ),
            "_available_terminals": [
                "GetRed", "GetGreen", "GetBlue", "GetGray"
            ],
            "_available_primitives": {
                "feature_extraction": [
                    "histogram_features(Channel) -> FeatureVector",
                    "hog_features(Channel) -> FeatureVector", 
                    "lbp_features(Channel) -> FeatureVector",
                    "sift_features(Channel) -> FeatureVector",
                    "sobel_features(Channel) -> FeatureVector",
                    "gabor_features(Channel, Theta, Frequency) -> FeatureVector",
                    "gaussian_features(Channel, Sigma) -> FeatureVector",
                    "concat_images(Channel, Channel) -> FeatureVector"
                ],
                "classification": [
                    "rf_classification(FeatureVector, Trees, Depth) -> Prediction",
                    "erf_classification(FeatureVector, Trees, Depth) -> Prediction",
                    "lr_classification(FeatureVector) -> Prediction",
                    "svm_classification(FeatureVector) -> Prediction"
                ],
                "filters": [
                    "mean_filter(Channel) -> Channel",
                    "gaussian_filter(Channel, Sigma) -> Channel",
                    "sobel_filter(Channel) -> Channel",
                    "laplacian_filter(Channel) -> Channel"
                ],
                "ensemble": [
                    "sum_prediction_2(Prediction, Prediction) -> Prediction",
                    "sum_prediction_3(Prediction, Prediction, Prediction) -> Prediction"
                ]
            },
            "_parameter_ranges": {
                "Trees": "50-1000 (step 50)",
                "Depth": "10-100 (step 10)", 
                "Sigma": "1-3",
                "Theta": "0 to 7*pi/8 (multiples of pi/8)",
                "Frequency": "pi/8 to pi/2 (multiples of pi/8)"
            },
            "_previous_generation_results": previous_results or [],
            "individuals": [
                "# Replace this with your S-expressions",
                "# Example: rf_classification(hog_features(GetRed), 100, 50)"
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        return json_path
    
    def _load_and_parse_population(self, json_path: str) -> Tuple[List[gp.PrimitiveTree], List[str]]:
        """
        Load and parse population from JSON file.
        
        Args:
            json_path: Path to population.json
            
        Returns:
            Tuple of (list of valid individuals, list of error messages)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'individuals' not in data:
            return [], ["JSON file missing 'individuals' array"]
        
        individuals = []
        errors = []
        
        for idx, expr_str in enumerate(data['individuals']):
            # Skip comments (strings starting with #)
            if isinstance(expr_str, str) and expr_str.strip().startswith('#'):
                continue
            
            if not isinstance(expr_str, str):
                errors.append(f"Individual {idx}: Expected string, got {type(expr_str).__name__}")
                continue
            
            # Parse the S-expression
            ind, parse_error = self.parser.parse(expr_str)
            
            if parse_error:
                errors.append(f"Individual {idx}: {parse_error}\n  Expression: {expr_str[:100]}...")
                continue
            
            # Validate structure
            is_valid, validation_error = self.parser.validate_structure(ind)
            
            if not is_valid:
                errors.append(f"Individual {idx}: {validation_error}\n  Expression: {expr_str[:100]}...")
                continue
            
            # Store the original expression string for reference
            ind._original_expr = expr_str
            individuals.append(ind)
        
        return individuals, errors
    
    def _wait_for_user_input(self, json_path: str, gen_num: int) -> str:
        """
        Wait for user to edit the population file and confirm.
        
        Args:
            json_path: Path to the population.json file
            gen_num: Current generation number
            
        Returns:
            User's choice: 'continue', 'quit', or 'retry'
        """
        print("\n" + "=" * 70)
        print(f"GENERATION {gen_num} - AWAITING USER INPUT")
        print("=" * 70)
        print(f"\nPopulation file created: {json_path}")
        print("\nPlease edit the 'individuals' array in this file with your S-expressions.")
        print("Each individual should be a valid GP tree expression.")
        print("\nExample expressions:")
        print("  - rf_classification(hog_features(GetRed), 100, 50)")
        print("  - lr_classification(histogram_features(gaussian_filter(GetGray, 2)))")
        print("  - sum_prediction_2(rf_classification(hog_features(GetRed), 200, 30), ")
        print("                     lr_classification(lbp_features(GetBlue)))")
        print("\n" + "-" * 70)
        
        while True:
            print("\nOptions:")
            print("  [c] Continue - Read population file and evaluate")
            print("  [q] Quit     - Save progress and generate summary chart")
            print("  [r] Refresh  - Show this menu again")
            
            try:
                choice = input("\nEnter your choice [c/q/r]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\nInterrupt received. Saving progress...")
                return 'quit'
            
            if choice == 'c' or choice == 'continue':
                return 'continue'
            elif choice == 'q' or choice == 'quit':
                confirm = input("Are you sure you want to quit? [y/n]: ").strip().lower()
                if confirm == 'y' or confirm == 'yes':
                    return 'quit'
            elif choice == 'r' or choice == 'refresh':
                continue
            else:
                print("Invalid choice. Please enter 'c', 'q', or 'r'.")
    
    def _apply_results_to_population(self, population: List, results: List[EvaluationResult]):
        """
        Apply evaluation results from workers back to the population.
        
        Args:
            population: List of DEAP individuals
            results: List of EvaluationResult objects
        """
        results_by_idx = {r.individual_idx: r for r in results}
        
        for idx, ind in enumerate(population):
            if idx in results_by_idx:
                result = results_by_idx[idx]
                ind.train_acc = result.train_acc
                ind.val_acc = result.val_acc
                ind.test_acc = result.test_acc
                ind.fitness.values = (result.val_acc,)
    
    def save_generation(self, population: List, gen_num: int) -> List[Dict]:
        """
        Save generation statistics and results.
        
        Returns:
            List of result dictionaries for each individual (for next gen template)
        """
        gen_dir = os.path.join(self.base_dir, f"gen_{gen_num}")
        os.makedirs(gen_dir, exist_ok=True)
        
        print(f"\nSaving Generation {gen_num} statistics...")
        
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        depths = []
        sizes = []
        individual_results = []
        
        for i, ind in enumerate(population):
            train_acc = getattr(ind, "train_acc", 0.0)
            val_acc = getattr(ind, "val_acc", 0.0)
            test_acc = getattr(ind, "test_acc", 0.0)
            
            result_info = {
                "id": i,
                "train_acc": round(float(train_acc), 4),
                "val_acc": round(float(val_acc), 4),
                "test_acc": round(float(test_acc), 4),
                "expression": str(ind),
                "depth": ind.height,
                "size": len(ind)
            }
            individual_results.append(result_info)
            
            # Track global best
            if val_acc > self.best_fitness_overall:
                self.best_fitness_overall = val_acc
                self.best_individual_overall = {
                    "generation": gen_num,
                    **result_info
                }
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            depths.append(ind.height)
            sizes.append(len(ind))
            
            # Draw tree visualization
            tree_dir = os.path.join(gen_dir, f"tree_{i}")
            os.makedirs(tree_dir, exist_ok=True)
            try:
                gp_draw.draw_tree(ind, filename=os.path.join(tree_dir, "tree_viz.png"))
            except Exception as e:
                print(f"  Warning: Failed to draw tree {i}: {e}")
            
            # Save individual results
            with open(os.path.join(tree_dir, "results.json"), "w") as f:
                json.dump(result_info, f, indent=4)

        # Calculate and save generation summary
        gen_summary = {
            "generation": gen_num,
            "population_size": len(population),
            "avg_train_accuracy": float(np.mean(train_accuracies)),
            "max_train_accuracy": float(np.max(train_accuracies)),
            "min_train_accuracy": float(np.min(train_accuracies)),
            "std_train_accuracy": float(np.std(train_accuracies)),
            
            "avg_val_accuracy": float(np.mean(val_accuracies)),
            "max_val_accuracy": float(np.max(val_accuracies)),
            "min_val_accuracy": float(np.min(val_accuracies)),
            "std_val_accuracy": float(np.std(val_accuracies)),
            
            "avg_test_accuracy": float(np.mean(test_accuracies)),
            "max_test_accuracy": float(np.max(test_accuracies)),
            "min_test_accuracy": float(np.min(test_accuracies)),
            "std_test_accuracy": float(np.std(test_accuracies)),
            
            "avg_depth": float(np.mean(depths)),
            "avg_size": float(np.mean(sizes)),
            "individual_results": individual_results
        }
        
        self.all_generations_stats.append(gen_summary)
        
        with open(os.path.join(gen_dir, "generation_summary.json"), "w") as f:
            json.dump(gen_summary, f, indent=4)
        
        print(f"  Avg Val: {gen_summary['avg_val_accuracy']:.4f}, "
              f"Max Val: {gen_summary['max_val_accuracy']:.4f}, "
              f"Avg Test: {gen_summary['avg_test_accuracy']:.4f}")
        
        return individual_results
    
    def save_experiment_summary(self):
        """Save final experiment summary and progress plot."""
        print("\n" + "=" * 70)
        print("SAVING EXPERIMENT SUMMARY")
        print("=" * 70)
        
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_type": "interactive",
            "total_generations_completed": len(self.all_generations_stats),
            "n_workers": self.n_workers,
            "best_individual": self.best_individual_overall,
            "generations_stats": self.all_generations_stats
        }
        
        with open(os.path.join(self.base_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nExperiment summary saved to: {self.base_dir}/experiment_summary.json")
        
        # Generate progress plot
        if len(self.all_generations_stats) > 0:
            self._plot_progress()
        
        if self.best_individual_overall:
            print(f"\n{'Best Individual':=^70}")
            print(f"  Generation: {self.best_individual_overall['generation']}")
            print(f"  Validation Accuracy: {self.best_individual_overall['val_acc']:.4f}")
            print(f"  Test Accuracy: {self.best_individual_overall['test_acc']:.4f}")
            print(f"  Expression: {self.best_individual_overall['expression'][:100]}...")
    
    def _plot_progress(self):
        """Generate and save progress plot."""
        try:
            gens = [s["generation"] for s in self.all_generations_stats]
            avg_train = [s["avg_train_accuracy"] for s in self.all_generations_stats]
            max_train = [s["max_train_accuracy"] for s in self.all_generations_stats]
            avg_val = [s["avg_val_accuracy"] for s in self.all_generations_stats]
            max_val = [s["max_val_accuracy"] for s in self.all_generations_stats]
            avg_test = [s["avg_test_accuracy"] for s in self.all_generations_stats]
            max_test = [s["max_test_accuracy"] for s in self.all_generations_stats]
            
            plt.figure(figsize=(14, 7))
            
            # Plot with markers for better visibility
            plt.plot(gens, avg_train, label="Avg Train", linestyle="--", marker="o", 
                     alpha=0.7, markersize=8)
            plt.plot(gens, max_train, label="Max Train", linestyle="-", marker="o",
                     markersize=8)
            plt.plot(gens, avg_val, label="Avg Val", linestyle="--", marker="x", 
                     alpha=0.7, markersize=8)
            plt.plot(gens, max_val, label="Max Val", linestyle="-", marker="x",
                     markersize=8)
            plt.plot(gens, avg_test, label="Avg Test", linestyle=":", marker="s", 
                     alpha=0.5, markersize=8)
            plt.plot(gens, max_test, label="Max Test", linestyle="-.", marker="s",
                     alpha=0.7, markersize=8)
            
            plt.title(f"Interactive Experiment Progress: {self.experiment_name}")
            plt.xlabel("Generation")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Set x-axis to integers only
            plt.xticks(gens)
            
            # Add generation markers
            for g in gens:
                plt.axvline(x=g, color='gray', linestyle=':', alpha=0.2)
            
            plt.tight_layout()
            plot_path = os.path.join(self.base_dir, "progress_plot.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            print(f"Progress plot saved to: {plot_path}")
            
        except Exception as e:
            print(f"Warning: Failed to generate progress plot: {e}")
    
    def run(self, max_generations: int = None):
        """
        Run the interactive GP experiment.
        
        Workflow:
        1. Create population template for gen 0
        2. Wait for user to edit and confirm
        3. Parse and validate population
        4. Evaluate population in parallel
        5. Save results
        6. Create template for next generation
        7. Repeat until user quits or max_generations reached
        
        Args:
            max_generations: Optional maximum number of generations (None = unlimited)
        """
        print("\n" + "=" * 70)
        print(f"INTERACTIVE GP EXPERIMENT: {self.experiment_name}")
        print("=" * 70)
        
        # Detect available workers
        if self.n_workers is None:
            self.n_workers = get_available_workers()
        
        print(f"\nConfiguration:")
        print(f"  Workers: {self.n_workers}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Max Generations: {max_generations or 'unlimited'}")
        print(f"  Output Directory: {self.base_dir}")
        
        # Start heartbeat (writes alive signal every 60s)
        start_heartbeat(interval=60, log_dir=self.base_dir, include_memory=True)
        
        # Create parallel evaluator
        self.evaluator = ParallelEvaluator(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
            batch_size=self.batch_size,
            pset_context=self.pset.context,
            n_workers=self.n_workers
        )
        
        previous_results = None
        gen_num = 0
        
        try:
            while max_generations is None or gen_num < max_generations:
                # Create population template
                json_path = self._create_population_template(gen_num, previous_results)
                
                # Wait for user input
                while True:
                    user_choice = self._wait_for_user_input(json_path, gen_num)
                    
                    if user_choice == 'quit':
                        print("\nUser requested quit. Saving progress...")
                        self.save_experiment_summary()
                        return
                    
                    # Try to load and parse population
                    print(f"\nParsing population from: {json_path}")
                    population, errors = self._load_and_parse_population(json_path)
                    
                    if errors:
                        print("\n" + "!" * 70)
                        print("PARSING ERRORS DETECTED")
                        print("!" * 70)
                        for error in errors:
                            print(f"\n  ✗ {error}")
                        print("\n" + "-" * 70)
                        print(f"Successfully parsed: {len(population)} individuals")
                        print(f"Errors: {len(errors)}")
                        print("\nPlease fix the errors in the JSON file and try again.")
                        continue
                    
                    if len(population) == 0:
                        print("\n" + "!" * 70)
                        print("NO VALID INDIVIDUALS FOUND")
                        print("!" * 70)
                        print("\nPlease add at least one valid S-expression to the 'individuals' array.")
                        continue
                    
                    # Successfully parsed
                    print(f"\n✓ Successfully parsed {len(population)} individuals")
                    break
                
                start_time = time.time()
                
                # Evaluate population
                print(f"\n--- Evaluating Generation {gen_num} ({len(population)} individuals) ---")
                self.logger.info("Evaluating generation %d  |  %d individuals",
                                 gen_num, len(population))
                
                with TimingContext(f"Generation {gen_num} evaluation", self.logger):
                    results = self.evaluator.evaluate_population(population, gen_num, self.base_dir)
                    self._apply_results_to_population(population, results)
                
                gen_time = time.time() - start_time
                self.logger.info("Generation %d completed in %.2fs", gen_num, gen_time)
                
                # Save generation results
                previous_results = self.save_generation(population, gen_num)
                
                gen_num += 1
                self.current_generation = gen_num
                
            # Reached max generations
            print(f"\nReached maximum generations ({max_generations})")
            self.save_experiment_summary()
            
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt received. Saving progress...")
            self.save_experiment_summary()
            
        finally:
            stop_heartbeat()
            if self.evaluator:
                self.evaluator.shutdown()


def run_interactive_experiment(experiment_name: str,
                               X_train, y_train, X_val, y_val, X_test, y_test,
                               pset, batch_size=32, n_workers=None,
                               max_generations=None, output_dir="experiments"):
    """
    Convenience function to run an interactive GP experiment.
    
    Args:
        experiment_name: Name of the experiment
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        pset: DEAP primitive set
        batch_size: Batch size for evaluation
        n_workers: Number of workers (default: auto-detect)
        max_generations: Maximum number of generations (None = unlimited)
        output_dir: Output directory for results
        
    Returns:
        InteractiveExperimentRunner instance with results
    """
    runner = InteractiveExperimentRunner(
        experiment_name=experiment_name,
        output_dir=output_dir,
        n_workers=n_workers
    )
    
    runner.setup_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    runner.setup_gp(pset)
    runner.run(max_generations=max_generations)
    
    return runner
