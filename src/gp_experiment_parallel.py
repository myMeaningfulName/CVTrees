"""
Parallel Experiment Runner for GP Evolution.

This module provides a parallelized version of ExperimentRunner that distributes
tree evaluations across multiple CPU cores using multiprocessing.

Key features:
- Complete process isolation: each worker has its own context
- Dynamic work distribution: workers grab new trees as they finish
- SLURM-aware: auto-detects available CPUs from SLURM_CPUS_PER_TASK
- Main process handles reproduction after all evaluations complete
- Results are saved by workers to avoid bottlenecks
"""

import os
import json
import time
import numpy as np
from deap import base, creator, tools, gp
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src import gp_setup, gp_ops, gp_types, gp_context, gp_draw, gp_utils
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


class ParallelExperimentRunner:
    """
    Parallelized experiment runner for GP evolution.
    
    Evaluates population members in parallel using a process pool.
    Each worker process has completely isolated state to prevent any
    memory sharing or information leakage between tree evaluations.
    
    The main process handles:
    - Population initialization
    - Selection, crossover, and mutation (reproduction)
    - Generation statistics and experiment summary
    
    Worker processes handle:
    - Tree evaluation (train, val, test)
    - Saving individual results to filesystem
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "experiments",
                 n_workers: int = None):
        """
        Initialize the parallel experiment runner.
        
        Args:
            experiment_name: Name of the experiment (used for output directory)
            output_dir: Base output directory
            n_workers: Number of worker processes (default: auto-detect)
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.base_dir = os.path.join(output_dir, experiment_name)
        self.n_workers = n_workers  # Will be set in run() if None
        
        if os.path.exists(self.base_dir):
            print(f"Warning: Experiment directory {self.base_dir} exists.")
        else:
            os.makedirs(self.base_dir)
        
        # Setup logging & heartbeat
        log_file = setup_experiment_logging(experiment_name, output_dir)
        self.logger = get_logger("experiment")
        self.logger.info("ParallelExperimentRunner created  |  experiment=%s", experiment_name)
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.batch_size = 32
        self.pset = None
        self.toolbox = None
        
        # Global stats tracking
        self.all_generations_stats = []
        self.best_individual_overall = None
        self.best_fitness_overall = -1.0
        
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
        
    def setup_gp(self, pset, pop_size=20, generations=10, 
                 crossover_prob=0.5, mutation_prob=0.2, max_depth=10):
        """
        Setup GP parameters and DEAP toolbox.
        
        Args:
            pset: DEAP primitive set
            pop_size: Population size
            generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            max_depth: Maximum tree depth (default: 10)
        """
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.pset = pset
        
        # Setup DEAP creator
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp_utils.gen_safe, pset=pset, 
                              min_=2, max_=max_depth, type_=gp_types.EnsembleOutput)
        self.toolbox.register("individual", tools.initIterate, 
                              creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, 
                              list, self.toolbox.individual)
        
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, 
                              expr=self.toolbox.expr_mut, pset=pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(
            key=lambda ind: ind.height, max_value=max_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(
            key=lambda ind: ind.height, max_value=max_depth))

    def _apply_results_to_population(self, population, results: list):
        """
        Apply evaluation results from workers back to the population.
        
        Args:
            population: List of DEAP individuals
            results: List of EvaluationResult objects
        """
        # Create lookup by index
        results_by_idx = {r.individual_idx: r for r in results}
        
        for idx, ind in enumerate(population):
            if idx in results_by_idx:
                result = results_by_idx[idx]
                ind.train_acc = result.train_acc
                ind.val_acc = result.val_acc
                ind.test_acc = result.test_acc
                ind.fitness.values = (result.val_acc,)

    def save_generation(self, population, gen_num: int):
        """
        Save generation statistics and update best individual tracking.
        
        Note: Individual results (results.json) are already saved by workers.
        This method handles generation-level statistics and tree visualizations.
        """
        gen_dir = os.path.join(self.base_dir, f"gen_{gen_num}")
        os.makedirs(gen_dir, exist_ok=True)
        
        self.logger.info("Saving generation %d statistics...", gen_num)
        
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        depths = []
        sizes = []
        
        for i, ind in enumerate(population):
            train_acc = getattr(ind, "train_acc", 0.0)
            val_acc = getattr(ind, "val_acc", 0.0)
            test_acc = getattr(ind, "test_acc", 0.0)
            
            # Track global best
            if val_acc > self.best_fitness_overall:
                self.best_fitness_overall = val_acc
                self.best_individual_overall = {
                    "generation": gen_num,
                    "id": i,
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "expression": str(ind),
                    "depth": ind.height,
                    "size": len(ind)
                }
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            depths.append(ind.height)
            sizes.append(len(ind))
            
            # Draw tree visualization (main process, not performance critical)
            tree_dir = os.path.join(gen_dir, f"tree_{i}")
            os.makedirs(tree_dir, exist_ok=True)
            try:
                gp_draw.draw_tree(ind, filename=os.path.join(tree_dir, "tree_viz.png"))
            except Exception as e:
                print(f"Failed to draw tree {i}: {e}")
            
            # Update results.json with depth and size (workers don't have the individual object)
            results_path = os.path.join(tree_dir, "results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path, "r") as f:
                        stats = json.load(f)
                    stats["depth"] = ind.height
                    stats["size"] = len(ind)
                    with open(results_path, "w") as f:
                        json.dump(stats, f, indent=4)
                except Exception:
                    pass

        # Calculate and save generation summary
        gen_summary = {
            "generation": gen_num,
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
            "avg_size": float(np.mean(sizes))
        }
        
        self.all_generations_stats.append(gen_summary)
        
        with open(os.path.join(gen_dir, "generation_summary.json"), "w") as f:
            json.dump(gen_summary, f, indent=4)
        
        self.logger.info("Gen %d: avg_val=%.4f  max_val=%.4f  avg_test=%.4f",
                         gen_num, gen_summary['avg_val_accuracy'],
                         gen_summary['max_val_accuracy'],
                         gen_summary['avg_test_accuracy'])
        log_memory(label=f"post-save-gen{gen_num}", logger=self.logger)

    def save_experiment_summary(self):
        """Save final experiment summary and progress plot."""
        self.logger.info("Saving experiment summary...")
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_generations": self.generations,
            "population_size": self.pop_size,
            "n_workers": self.n_workers,
            "total_trees_evaluated": (self.generations + 1) * self.pop_size,
            "best_individual": self.best_individual_overall,
            "generations_stats": self.all_generations_stats
        }
        
        with open(os.path.join(self.base_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        
        # Plot progress
        try:
            gens = [s["generation"] for s in self.all_generations_stats]
            avg_train = [s["avg_train_accuracy"] for s in self.all_generations_stats]
            max_train = [s["max_train_accuracy"] for s in self.all_generations_stats]
            avg_val = [s["avg_val_accuracy"] for s in self.all_generations_stats]
            max_val = [s["max_val_accuracy"] for s in self.all_generations_stats]
            avg_test = [s["avg_test_accuracy"] for s in self.all_generations_stats]
            
            plt.figure(figsize=(12, 6))
            plt.plot(gens, avg_train, label="Avg Train", linestyle="--", marker="o", alpha=0.7)
            plt.plot(gens, max_train, label="Max Train", linestyle="-", marker="o")
            plt.plot(gens, avg_val, label="Avg Val", linestyle="--", marker="x", alpha=0.7)
            plt.plot(gens, max_val, label="Max Val", linestyle="-", marker="x")
            plt.plot(gens, avg_test, label="Avg Test", linestyle=":", marker="s", alpha=0.5)
            
            plt.title(f"Experiment Progress: {self.experiment_name} ({self.n_workers} workers)")
            plt.xlabel("Generation")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.base_dir, "progress_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot summary: {e}")

    def run(self):
        """
        Run the parallel GP experiment.
        
        Workflow:
        1. Initialize population
        2. Create parallel evaluator with worker pool
        3. For each generation:
           a. Evaluate population in parallel (workers)
           b. Collect results and update fitness
           c. Save generation statistics (main process)
           d. Selection, crossover, mutation (main process)
        4. Save experiment summary
        5. Shutdown worker pool
        """
        self.logger.info("Starting Parallel Experiment: %s", self.experiment_name)
        
        # Detect available workers
        if self.n_workers is None:
            self.n_workers = get_available_workers()
        
        self.logger.info(
            "Configuration: Population=%d  Generations=%d  Workers=%d  "
            "CX=%.2f  MUT=%.2f  batch_size=%d",
            self.pop_size, self.generations, self.n_workers,
            self.cx_prob, self.mut_prob, self.batch_size)
        
        # Start heartbeat (writes alive signal every 60s)
        start_heartbeat(interval=60, log_dir=self.base_dir, include_memory=True)
        
        # Initialize population (main process)
        pop = self.toolbox.population(n=self.pop_size)
        self.logger.info("Initial population created (%d individuals)", len(pop))
        
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
        
        experiment_start = time.time()
        
        try:
            # Evaluate Generation 0
            self.logger.info("\n=== Generation 0 ===")
            
            with TimingContext("Generation 0 evaluation", self.logger):
                results = self.evaluator.evaluate_population(pop, 0, self.base_dir)
                self._apply_results_to_population(pop, results)
                self.save_generation(pop, 0)
            
            # Evolution loop
            for g in range(1, self.generations + 1):
                self.logger.info("\n=== Generation %d ===", g)
                gen_start = time.time()
                
                # Selection (main process)
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover (main process)
                cx_count = 0
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.random() < self.cx_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                        if hasattr(child1, "train_acc"): 
                            del child1.train_acc
                        if hasattr(child2, "train_acc"): 
                            del child2.train_acc
                        cx_count += 1
                
                # Mutation (main process)
                mut_count = 0
                for mutant in offspring:
                    if np.random.random() < self.mut_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                        if hasattr(mutant, "train_acc"): 
                            del mutant.train_acc
                        mut_count += 1
                
                # Find individuals that need evaluation
                invalid_indices = [i for i, ind in enumerate(offspring) 
                                   if not ind.fitness.valid]
                invalid_ind = [offspring[i] for i in invalid_indices]
                
                self.logger.info(
                    "Gen %d reproduction: %d crossovers, %d mutations, "
                    "%d need re-evaluation",
                    g, cx_count, mut_count, len(invalid_ind))
                
                if invalid_ind:
                    with TimingContext(f"Generation {g} evaluation", self.logger):
                        results = self.evaluator.evaluate_individuals(
                            invalid_ind, invalid_indices, g, self.base_dir
                        )
                        self._apply_results_to_population(offspring, results)
                
                # Copy over results for individuals that weren't re-evaluated
                for i, ind in enumerate(offspring):
                    if i not in invalid_indices:
                        # This individual was copied from previous generation
                        # Results already saved, just ensure tree_dir exists
                        tree_dir = os.path.join(self.base_dir, f"gen_{g}", f"tree_{i}")
                        os.makedirs(tree_dir, exist_ok=True)
                        
                        # Copy stats to results.json
                        stats = {
                            "generation": g,
                            "individual_id": i,
                            "train_accuracy": float(getattr(ind, "train_acc", 0.0)),
                            "validation_accuracy": float(getattr(ind, "val_acc", 0.0)),
                            "test_accuracy": float(getattr(ind, "test_acc", 0.0)),
                            "expression": str(ind),
                            "note": "copied from previous generation"
                        }
                        with open(os.path.join(tree_dir, "results.json"), "w") as f:
                            json.dump(stats, f, indent=4)
                
                pop[:] = offspring
                self.save_generation(pop, g)
                
                gen_time = time.time() - gen_start
                self.logger.info("Generation %d completed in %.2fs", g, gen_time)
            
            # Save final summary
            total_time = time.time() - experiment_start
            self.logger.info("Experiment completed in %.2fs (%.1f min)",
                             total_time, total_time / 60)
            self.save_experiment_summary()
            self.logger.info("Experiment Completed Successfully.")
            
        except Exception as exc:
            self.logger.critical("Experiment CRASHED", exc_info=True)
            log_memory(label="crash", logger=self.logger)
            # Still try to save what we have
            try:
                self.save_experiment_summary()
            except Exception:
                self.logger.error("Failed to save summary after crash", exc_info=True)
            raise
            
        finally:
            # Always cleanup the pool and heartbeat
            stop_heartbeat()
            if self.evaluator:
                self.evaluator.shutdown()


# Convenience function to run parallel experiments
def run_parallel_experiment(experiment_name: str, 
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            pset, pop_size=20, generations=10,
                            crossover_prob=0.5, mutation_prob=0.2,
                            batch_size=32, n_workers=None,
                            output_dir="experiments", max_depth=10):
    """
    Convenience function to run a parallel GP experiment.
    
    Args:
        experiment_name: Name of the experiment
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        pset: DEAP primitive set
        pop_size: Population size
        generations: Number of generations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        batch_size: Batch size for evaluation
        n_workers: Number of workers (default: auto-detect)
        output_dir: Output directory for results
        max_depth: Maximum tree depth (default: 10)
        
    Returns:
        ParallelExperimentRunner instance with results
    """
    runner = ParallelExperimentRunner(
        experiment_name=experiment_name,
        output_dir=output_dir,
        n_workers=n_workers
    )
    
    runner.setup_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    runner.setup_gp(pset, pop_size, generations, crossover_prob, mutation_prob, max_depth)
    runner.run()
    
    return runner
