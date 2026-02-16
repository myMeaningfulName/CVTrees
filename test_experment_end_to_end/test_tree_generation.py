#!/usr/bin/env python3
"""
Test script to generate random GP trees and visualize them.
This helps verify that the GP language is properly defined and generates legal trees.

A legal tree should follow this structure:
1. Input: Channel terminals (GetRed, GetGreen, GetBlue, GetGray) at the leaves
2. Image Filtering: Channel -> Channel (optional, can be stacked)
3. Feature Extraction: Channel -> FeatureVector
4. Feature Concatenation: FeatureVector x N -> FeatureVector (optional)
5. Cascade Classification: FeatureVector -> FeatureVector (optional)
6. Final Classification: FeatureVector -> Prediction
7. Ensemble Summation: Prediction x N -> Prediction (optional)

Run this script to generate N random trees and inspect them visually.

Usage:
    python test_tree_generation.py [options]

Options:
    --extended-ops       Enable extended filtering/feature extraction operations
    --extended-params    Enable extended parameterization of operations
    --num-trees N        Number of trees to generate (default: 10)
    --min-depth N        Minimum tree depth (default: 2)
    --max-depth N        Maximum tree depth (default: 5)
    --seed N             Random seed (default: 42)
    --output-dir DIR     Output directory (default: test_generated_trees)
    
Examples:
    # Basic mode (current behavior)
    python test_tree_generation.py
    
    # Extended operations only
    python test_tree_generation.py --extended-ops
    
    # Extended parameters only
    python test_tree_generation.py --extended-params
    
    # Both extended modes
    python test_tree_generation.py --extended-ops --extended-params
"""

import os
import sys
import random
import warnings
import importlib
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deap import base, creator, tools, gp
from src import gp_setup, gp_types, gp_utils, gp_draw

# Reload to pick up latest changes
importlib.reload(gp_utils)
importlib.reload(gp_setup)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and visualize random GP trees for image classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic mode (current behavior)
    python test_tree_generation.py
    
    # Extended operations only
    python test_tree_generation.py --extended-ops
    
    # Extended parameters only
    python test_tree_generation.py --extended-params
    
    # Both extended modes
    python test_tree_generation.py --extended-ops --extended-params
    
    # Custom configuration
    python test_tree_generation.py --extended-ops --num-trees 20 --max-depth 7
        """
    )
    
    # Mode flags (independent toggles)
    parser.add_argument('--extended-ops', action='store_true', default=False,
                        help='Enable extended filtering/feature extraction operations '
                             '(morphological ops, edge detectors, texture features, etc.)')
    parser.add_argument('--extended-params', action='store_true', default=False,
                        help='Enable extended parameterization of operations '
                             '(configurable pixels_per_cell, kernel sizes, etc.)')
    
    # Tree generation parameters
    parser.add_argument('--num-trees', type=int, default=10,
                        help='Number of trees to generate (default: 10)')
    parser.add_argument('--min-depth', type=int, default=2,
                        help='Minimum tree depth (default: 2)')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum tree depth (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='test_generated_trees',
                        help='Output directory for tree visualizations (default: test_generated_trees)')
    
    return parser.parse_args()

def validate_tree_structure(individual, pset):
    """
    Validates that a tree follows the expected legal structure:
    
    Input -> [Filtering]* -> Feature Extraction -> [Concat]? -> 
    [Cascade -> Concat?]* -> Classification -> Ensemble -> Output
    
    Key constraints:
    1. Must have Channel input terminals (GetRed, GetGreen, GetBlue, GetGray)
    2. Must have at least one classification head
    3. Must end with ensemble operation (ensemble_single, ensemble_sum_2, ensemble_sum_3)
    4. Should NOT have old-style sum_prediction (which allowed nesting)
    5. Ensemble/summation should ONLY appear at the root (not nested)
    
    Returns a dict with validation results.
    """
    expr_str = str(individual)
    issues = []
    
    # Check for degenerate patterns that should NOT exist anymore
    if "DefaultHOG" in expr_str:
        issues.append("Contains DefaultHOG terminal (bypasses Channel processing)")
    
    if "DefaultRF" in expr_str:
        issues.append("Contains DefaultRF terminal (trivial Prediction)")
    
    # Check for identity function bloat (should NOT exist anymore)
    if "identity_trees" in expr_str:
        issues.append("Contains identity_trees (unnecessary bloat)")
    if "identity_depth" in expr_str:
        issues.append("Contains identity_depth (unnecessary bloat)")
    if "identity_sigma" in expr_str:
        issues.append("Contains identity_sigma (unnecessary bloat)")
    
    # Check that tree terminates with Channel terminals
    channel_terminals = ["GetRed", "GetGreen", "GetBlue", "GetGray"]
    has_channel_input = any(t in expr_str for t in channel_terminals)
    if not has_channel_input:
        issues.append("No Channel input terminals (GetRed, GetGreen, GetBlue, GetGray)")
    
    # Check that tree has at least one classification head
    classifiers = ["rf_classification", "erf_classification", "lr_classification", "svm_classification"]
    has_classifier = any(c in expr_str for c in classifiers)
    if not has_classifier:
        issues.append("No classification head found")
    
    # Check that tree ends with ensemble operation (new structure)
    ensemble_ops = ["ensemble_single", "ensemble_sum_2", "ensemble_sum_3"]
    has_ensemble = any(e in expr_str for e in ensemble_ops)
    if not has_ensemble:
        issues.append("No ensemble output operation found (must end with ensemble_single/sum_2/sum_3)")
    
    # Check for OLD-style summation that allowed nesting (deprecated)
    old_summation = ["sum_prediction_2", "sum_prediction_3"]
    has_old_summation = any(s in expr_str for s in old_summation)
    if has_old_summation:
        issues.append("Contains deprecated sum_prediction (use ensemble_sum instead)")
    
    # Check that ensemble operation is at the root (first function in expression)
    # The expression format is: func(arg1, arg2, ...)
    root_is_ensemble = any(expr_str.startswith(e + "(") for e in ensemble_ops)
    if has_ensemble and not root_is_ensemble:
        issues.append("Ensemble operation is not at root (illegal: nested ensemble)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "depth": individual.height,
        "size": len(individual),
        "expression": expr_str
    }
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "depth": individual.height,
        "size": len(individual),
        "expression": expr_str
    }

def analyze_tree_components(expr_str, extended_ops=False, extended_params=False):
    """
    Analyzes and categorizes the components used in a tree.
    """
    components = {
        "channel_inputs": [],
        "filters": [],
        "extended_filters": [],
        "feature_extractors": [],
        "extended_feature_extractors": [],
        "parameterized_ops": [],
        "concatenation": [],
        "cascades": [],
        "classifiers": [],
        "ensemble": [],           # New ensemble operations (at root)
        "legacy_summation": []    # Deprecated sum_prediction (should not appear)
    }
    
    # Channel inputs
    for t in ["GetRed", "GetGreen", "GetBlue", "GetGray"]:
        if t in expr_str:
            components["channel_inputs"].append(t)
    
    # Basic filters
    basic_filters = ["mean_filter", "median_filter", "min_filter", "max_filter", 
                     "gaussian_filter", "laplacian_filter", "sobel_filter", "relu", 
                     "sqrt_op", "gaud_filter", "log1_filter", "log2_filter", 
                     "gabor_filter", "hog_filter", "lbp_filter", "add_max_pool", 
                     "sub_max_pool", "linear_combination"]
    for f in basic_filters:
        # Avoid matching parameterized versions
        if f in expr_str and f + "_param" not in expr_str:
            components["filters"].append(f)
    
    # Extended filters (only check if extended_ops is enabled)
    extended_filters = ["prewitt_filter", "scharr_filter", "canny_filter", 
                        "bilateral_filter", "clahe_filter", "erosion_filter",
                        "dilation_filter", "opening_filter", "closing_filter",
                        "gradient_filter", "tophat_filter", "blackhat_filter",
                        "sharpen_filter", "emboss_filter", "dog_filter",
                        "normalize_filter", "invert_filter", "threshold_filter",
                        "adaptive_threshold_filter"]
    for f in extended_filters:
        if f in expr_str and f + "_param" not in expr_str:
            components["extended_filters"].append(f)
    
    # Basic feature extractors
    basic_extractors = ["histogram_features", "hog_features", "lbp_features", 
                        "concat_images", "sift_features", "hog_image_features",
                        "lbp_image_features", "sobel_features", "gabor_features",
                        "gaussian_features", "gaud_features"]
    for e in basic_extractors:
        if e in expr_str and e + "_param" not in expr_str:
            components["feature_extractors"].append(e)
    
    # Extended feature extractors (only check if extended_ops is enabled)
    extended_extractors = ["orb_features", "brief_features", "color_histogram_features",
                           "glcm_features", "hu_moments_features", "zernike_features",
                           "edge_histogram_features", "statistical_features",
                           "fourier_features", "daisy_features"]
    for e in extended_extractors:
        if e in expr_str:
            components["extended_feature_extractors"].append(e)
    
    # Parameterized operations (only check if extended_params is enabled)
    param_ops = ["hog_filter_param", "hog_features_param", "hog_image_features_param",
                 "lbp_filter_param", "lbp_features_param", "lbp_image_features_param",
                 "mean_filter_param", "median_filter_param", "min_filter_param",
                 "max_filter_param", "erosion_filter_param", "dilation_filter_param",
                 "opening_filter_param", "closing_filter_param", "canny_filter_param",
                 "sift_features_param"]
    for p in param_ops:
        if p in expr_str:
            components["parameterized_ops"].append(p)
    
    # Concatenation
    for c in ["concat_features_2", "concat_features_3", "concat_features_4"]:
        if c in expr_str:
            components["concatenation"].append(c)
    
    # Cascades
    for c in ["cascade_rf", "cascade_erf", "cascade_lr", "cascade_svm"]:
        if c in expr_str:
            components["cascades"].append(c)
    
    # Classifiers
    for c in ["rf_classification", "erf_classification", "lr_classification", "svm_classification"]:
        if c in expr_str:
            components["classifiers"].append(c)
    
    # Ensemble operations (new structure - these should be at root)
    for e in ["ensemble_single", "ensemble_sum_2", "ensemble_sum_3"]:
        if e in expr_str:
            components["ensemble"].append(e)
    
    # Legacy summation (deprecated - should not appear in new trees)
    for s in ["sum_prediction_2", "sum_prediction_3"]:
        if s in expr_str:
            components["legacy_summation"].append(s)
    
    return components

def main():
    args = parse_args()
    
    random.seed(args.seed)
    
    # Determine mode suffix for output directory
    mode_suffix = ""
    if args.extended_ops:
        mode_suffix += "_ext_ops"
    if args.extended_params:
        mode_suffix += "_ext_params"
    
    output_dir = args.output_dir + mode_suffix if mode_suffix else args.output_dir
    
    print("=" * 80)
    print("GP TREE GENERATION TEST")
    print("=" * 80)
    print(f"\nMode Configuration:")
    print(f"  Extended Operations: {'ENABLED' if args.extended_ops else 'DISABLED'}")
    print(f"  Extended Parameters: {'ENABLED' if args.extended_params else 'DISABLED'}")
    print(f"\nGenerating {args.num_trees} random trees with depth range [{args.min_depth}, {args.max_depth}]")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup DEAP with mode flags
    pset = gp_setup.create_primitive_set(
        extended_ops=args.extended_ops,
        extended_params=args.extended_params
    )
    
    # Print primitive set statistics
    pset_stats = gp_setup.count_primitives(pset)
    print(f"Primitive Set Statistics:")
    print(f"  Primitives: {pset_stats['primitives']}")
    print(f"  Terminals: {pset_stats['terminals']}")
    print(f"  Total: {pset_stats['total']}")
    print()
    
    # Create fitness and individual types (handle re-runs)
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    # Root type is now EnsembleOutput to enforce proper tree structure
    toolbox.register("expr", gp_utils.gen_safe, pset=pset, min_=args.min_depth, max_=args.max_depth, type_=gp_types.EnsembleOutput)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    
    # Generate trees
    valid_count = 0
    invalid_count = 0
    
    # Track component usage for summary
    component_usage = {
        "extended_filters_used": 0,
        "extended_extractors_used": 0,
        "parameterized_ops_used": 0
    }
    
    print("-" * 80)
    print("GENERATED TREES:")
    print("-" * 80)
    
    for i in range(args.num_trees):
        try:
            individual = toolbox.individual()
            validation = validate_tree_structure(individual, pset)
            components = analyze_tree_components(
                validation["expression"],
                extended_ops=args.extended_ops,
                extended_params=args.extended_params
            )
            
            # Track usage
            if components["extended_filters"]:
                component_usage["extended_filters_used"] += 1
            if components["extended_feature_extractors"]:
                component_usage["extended_extractors_used"] += 1
            if components["parameterized_ops"]:
                component_usage["parameterized_ops_used"] += 1
            
            print(f"\n[Tree {i}]")
            print(f"  Depth: {validation['depth']}, Size: {validation['size']}")
            print(f"  Valid: {'✓' if validation['valid'] else '✗'}")
            
            if not validation['valid']:
                print(f"  Issues: {validation['issues']}")
                invalid_count += 1
            else:
                valid_count += 1
            
            print(f"  Expression: {validation['expression'][:100]}{'...' if len(validation['expression']) > 100 else ''}")
            print(f"  Components:")
            print(f"    - Inputs: {components['channel_inputs']}")
            print(f"    - Filters: {components['filters']}")
            if args.extended_ops and components['extended_filters']:
                print(f"    - Extended Filters: {components['extended_filters']}")
            print(f"    - Feature Extractors: {components['feature_extractors']}")
            if args.extended_ops and components['extended_feature_extractors']:
                print(f"    - Extended Extractors: {components['extended_feature_extractors']}")
            if args.extended_params and components['parameterized_ops']:
                print(f"    - Parameterized Ops: {components['parameterized_ops']}")
            print(f"    - Concatenation: {components['concatenation']}")
            print(f"    - Cascades: {components['cascades']}")
            print(f"    - Classifiers: {components['classifiers']}")
            print(f"    - Ensemble (root): {components['ensemble']}")
            if components['legacy_summation']:
                print(f"    - [DEPRECATED] Legacy Summation: {components['legacy_summation']}")
            
            # Save tree visualization
            try:
                tree_file = os.path.join(output_dir, f"tree_{i}.png")
                gp_draw.draw_tree(individual, filename=tree_file)
            except Exception as e:
                print(f"  [Warning] Could not draw tree: {e}")
            
            # Save tree expression to text file
            expr_file = os.path.join(output_dir, f"tree_{i}.txt")
            with open(expr_file, "w") as f:
                f.write(f"Tree {i}\n")
                f.write(f"Mode: extended_ops={args.extended_ops}, extended_params={args.extended_params}\n")
                f.write(f"Depth: {validation['depth']}\n")
                f.write(f"Size: {validation['size']}\n")
                f.write(f"Valid: {validation['valid']}\n")
                if validation['issues']:
                    f.write(f"Issues: {validation['issues']}\n")
                f.write(f"\nExpression:\n{validation['expression']}\n")
                f.write(f"\nComponents:\n")
                for key, value in components.items():
                    if value:  # Only write non-empty components
                        f.write(f"  {key}: {value}\n")
                    
        except Exception as e:
            print(f"\n[Tree {i}] GENERATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            invalid_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Mode: extended_ops={args.extended_ops}, extended_params={args.extended_params}")
    print(f"Total trees generated: {args.num_trees}")
    print(f"Valid trees: {valid_count} ({100*valid_count/args.num_trees:.1f}%)")
    print(f"Invalid trees: {invalid_count} ({100*invalid_count/args.num_trees:.1f}%)")
    
    # Extended mode usage statistics
    if args.extended_ops:
        print(f"\nExtended Operations Usage:")
        print(f"  Trees using extended filters: {component_usage['extended_filters_used']} ({100*component_usage['extended_filters_used']/args.num_trees:.1f}%)")
        print(f"  Trees using extended extractors: {component_usage['extended_extractors_used']} ({100*component_usage['extended_extractors_used']/args.num_trees:.1f}%)")
    
    if args.extended_params:
        print(f"\nParameterized Operations Usage:")
        print(f"  Trees using parameterized ops: {component_usage['parameterized_ops_used']} ({100*component_usage['parameterized_ops_used']/args.num_trees:.1f}%)")
    
    print(f"\nTree visualizations saved to: {output_dir}/")
    print()
    
    # Print primitive set info for debugging
    print("-" * 80)
    print("PRIMITIVE SET INFO:")
    print("-" * 80)
    print(f"\nTerminals by type:")
    for type_, terms in pset.terminals.items():
        term_names = [getattr(t, 'name', str(t)) for t in terms]
        print(f"  {type_.__name__}: {term_names}")
    
    print(f"\nPrimitives by return type:")
    for type_, prims in pset.primitives.items():
        print(f"  {type_.__name__}:")
        for p in prims:
            arg_names = [a.__name__ for a in p.args]
            print(f"    - {p.name}({', '.join(arg_names)})")
    
    # Save configuration to output directory
    config_file = os.path.join(output_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write("GP Tree Generation Configuration\n")
        f.write("=" * 40 + "\n")
        f.write(f"extended_ops: {args.extended_ops}\n")
        f.write(f"extended_params: {args.extended_params}\n")
        f.write(f"num_trees: {args.num_trees}\n")
        f.write(f"min_depth: {args.min_depth}\n")
        f.write(f"max_depth: {args.max_depth}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"\nPrimitive Set Statistics:\n")
        f.write(f"  Primitives: {pset_stats['primitives']}\n")
        f.write(f"  Terminals: {pset_stats['terminals']}\n")
        f.write(f"  Total: {pset_stats['total']}\n")


if __name__ == "__main__":
    main()
