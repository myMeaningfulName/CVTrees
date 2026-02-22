import os
import json
import time
import numpy as np
from deap import base, creator, tools, gp, algorithms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import shutil

# Assuming these are in the path or relative
from src import gp_setup, gp_ops, gp_types, gp_context, gp_draw, gp_utils

class ExperimentRunner:
    def __init__(self, experiment_name, output_dir="experiments"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.base_dir = os.path.join(output_dir, experiment_name)
        
        # Clean start if exists? Or append? Let's clean for now to avoid confusion
        if os.path.exists(self.base_dir):
            print(f"Warning: Experiment directory {self.base_dir} exists.")
        else:
            os.makedirs(self.base_dir)
            
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
        
    def setup_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        
    def _make_image_iterator(self, X):
        """
        Creates a factory for an iterator that yields Batches of images.
        X is expected to be (N, 3, H, W).
        
        Always normalizes to [0, 1] float32 so that downstream filters and
        feature extractors (histogram, color_histogram, etc.) operate on a
        consistent range.
        """
        batch_size = self.batch_size
        def iterator():
            n_samples = len(X)
            for i in range(0, n_samples, batch_size):
                batch_data = X[i:i+batch_size]
                # gp_ops expects List[np.ndarray] for channels [R, G, B]
                # batch_data is (B, 3, H, W)
                r = batch_data[:, 0, :, :].astype(np.float32)
                g = batch_data[:, 1, :, :].astype(np.float32)
                b = batch_data[:, 2, :, :].astype(np.float32)
                # Normalize to [0, 1] if data is in [0, 255] range
                if r.max() > 1.0:
                    r = r / 255.0
                    g = g / 255.0
                    b = b / 255.0
                yield gp_types.Batch([r, g, b])
        return iterator

    def setup_gp(self, pset, pop_size=20, generations=10, crossover_prob=0.5, mutation_prob=0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.pset = pset
        
        # Setup DEAP
        # We need to ensure these are created globally for pickle compatibility if using multiprocessing,
        # but for single thread this is fine.
        if hasattr(creator, "FitnessMax"): del creator.FitnessMax
        if hasattr(creator, "Individual"): del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5, type_=gp_types.Prediction)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("compile", self._compile_lazy)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))
        self.toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))

    def _compile_lazy(self, expr):
        code = str(expr)
        code = f"lambda: {code}"
        return eval(code, self.pset.context, {})

    def _evaluate_pipeline(self, individual, X, y, mode=gp_context.ExecutionMode.TRAIN):
        """
        Evaluates an individual on a dataset X, y.
        """
        try:
            # 1. Compile (Lazy)
            func = self.toolbox.compile(expr=individual)
            
            # 2. Setup Context
            gp_context.context.reset(mode)
            if mode == gp_context.ExecutionMode.TRAIN:
                gp_context.context.set_train_labels(y)
            
            # 3. Setup Data
            iterator_factory = self._make_image_iterator(X)
            input_image = gp_types.Image(iterator_factory)
            gp_ops.set_image(input_image)
            
            # 4. Execute
            pipeline = func()
            
            preds = []
            for batch in pipeline:
                preds.append(batch.data)
                
            if not preds:
                return 0.0
                
            preds = np.vstack(preds)
            y_pred = np.argmax(preds, axis=1)
            acc = accuracy_score(y, y_pred)
            return acc
            
        except Exception as e:
            # print(f"Eval failed: {e}")
            return 0.0

    def _evaluate_and_store_stats(self, ind):
        """
        Runs the full evaluation cycle: Train -> Val -> Test.
        Stores results in the individual object.
        Sets fitness based on Validation Accuracy.
        """
        # 1. Train (Fit models)
        # We must run on train set to fit the classifiers inside the tree
        train_acc = self._evaluate_pipeline(ind, self.X_train, self.y_train, mode=gp_context.ExecutionMode.TRAIN)
        
        # 2. Validation (Fitness)
        # We use this for selection
        val_acc = self._evaluate_pipeline(ind, self.X_val, self.y_val, mode=gp_context.ExecutionMode.EVAL)
        
        # 3. Test (Reporting)
        # Purely for retrospect
        test_acc = self._evaluate_pipeline(ind, self.X_test, self.y_test, mode=gp_context.ExecutionMode.EVAL)
        
        # Store stats on the individual
        ind.train_acc = train_acc
        ind.val_acc = val_acc
        ind.test_acc = test_acc
        
        # Set Fitness (Validation Accuracy)
        ind.fitness.values = (val_acc,)
        
        return val_acc

    def save_generation(self, population, gen_num):
        gen_dir = os.path.join(self.base_dir, f"gen_{gen_num}")
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
            
        print(f"Saving Generation {gen_num} to {gen_dir}...")
        
        gen_stats = {
            "generation": gen_num,
            "population_size": len(population),
            "individuals": []
        }
        
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        depths = []
        sizes = []
        
        for i, ind in enumerate(population):
            tree_dir = os.path.join(gen_dir, f"tree_{i}")
            if not os.path.exists(tree_dir):
                os.makedirs(tree_dir)
                
            # Retrieve stored stats (calculated during evaluation)
            train_acc = getattr(ind, "train_acc", 0.0)
            val_acc = getattr(ind, "val_acc", 0.0)
            test_acc = getattr(ind, "test_acc", 0.0)
            
            # Track global best (based on Validation)
            if val_acc > self.best_fitness_overall:
                self.best_fitness_overall = val_acc
                self.best_individual_overall = {
                    "generation": gen_num,
                    "id": i,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "expression": str(ind),
                    "depth": ind.height,
                    "size": len(ind)
                }
            
            stats = {
                "generation": gen_num,
                "individual_id": i,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "test_accuracy": test_acc,
                "depth": ind.height,
                "size": len(ind),
                "expression": str(ind)
            }
            
            gen_stats["individuals"].append(stats)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            depths.append(ind.height)
            sizes.append(len(ind))
            
            # 3. Save JSON
            with open(os.path.join(tree_dir, "results.json"), "w") as f:
                json.dump(stats, f, indent=4)
                
            # 4. Save Image
            try:
                gp_draw.draw_tree(ind, filename=os.path.join(tree_dir, "tree_viz.png"))
            except Exception as e:
                print(f"Failed to draw tree {i}: {e}")

        # Calculate Generation Statistics
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
        
        # Save Generation Summary
        with open(os.path.join(gen_dir, "generation_summary.json"), "w") as f:
            json.dump(gen_summary, f, indent=4)

    def save_experiment_summary(self):
        print("Saving Experiment Summary...")
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_generations": self.generations,
            "population_size": self.pop_size,
            "total_trees_evaluated": (self.generations + 1) * self.pop_size,
            "best_individual": self.best_individual_overall,
            "generations_stats": self.all_generations_stats
        }
        
        with open(os.path.join(self.base_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
            
        # Plotting
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
            
            plt.title(f"Experiment Progress: {self.experiment_name}")
            plt.xlabel("Generation")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.base_dir, "progress_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot summary: {e}")

    def run(self):
        print(f"Starting Experiment: {self.experiment_name}")
        print(f"Population: {self.pop_size}, Generations: {self.generations}")
        
        # 1. Initialize Population
        pop = self.toolbox.population(n=self.pop_size)
        
        # 2. Evaluate Initial Population
        print("Evaluating Generation 0...")
        for ind in pop:
            self._evaluate_and_store_stats(ind)
            
        # Save Gen 0
        self.save_generation(pop, 0)
        
        # 3. Evolution Loop
        for g in range(1, self.generations + 1):
            print(f"\n--- Generation {g} ---")
            
            # Select (Based on Fitness = Validation Accuracy)
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Mate & Mutate
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    # Invalidate stats
                    if hasattr(child1, "train_acc"): del child1.train_acc
                    if hasattr(child2, "train_acc"): del child2.train_acc

            for mutant in offspring:
                if np.random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
                    if hasattr(mutant, "train_acc"): del mutant.train_acc
            
            # Evaluate Invalid
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"Evaluating {len(invalid_ind)} individuals...")
            
            for ind in invalid_ind:
                self._evaluate_and_store_stats(ind)
                
            pop[:] = offspring
            
            # Save Generation
            self.save_generation(pop, g)
            
        # Save Final Summary
        self.save_experiment_summary()
        print("Experiment Completed.")
