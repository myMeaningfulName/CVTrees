# Interactive GP Experiment Module

## Overview

The `gp_experiment_interactive` module provides an interactive version of the GP experiment runner where **you control the population manually** instead of using automatic GP operations (crossover/mutation).

This is useful for:
- **Manual experimentation**: Test specific tree structures you design
- **Guided search**: Apply domain knowledge to evolve better solutions
- **Debugging**: Understand how specific tree structures perform
- **Hybrid approaches**: Combine manual design with automated evaluation

## Key Features

1. **JSON-based population input**: Define individuals as S-expressions in a JSON file
2. **Interactive workflow**: Script waits for your input between generations
3. **Robust parsing**: Helpful error messages when syntax is incorrect
4. **Graceful exit**: Quit anytime and get a progress chart
5. **Parallel evaluation**: Same efficient parallel evaluation as `gp_experiment_parallel`

## Usage

### Running the CRC Test Script

```bash
python test_interactive_crc_experiment.py
```

On SLURM:
```bash
#SBATCH --cpus-per-task=8
srun python test_interactive_crc_experiment.py
```

### Workflow

1. **Script creates template**: `experiments/<name>/gen_0/population.json`
2. **You edit the file**: Add your S-expressions to the `individuals` array
3. **Signal ready**: Enter `c` in terminal to continue
4. **Script evaluates**: Population evaluated in parallel
5. **Results saved**: See results in `gen_0/` folder
6. **Next generation**: Script creates `gen_1/population.json`
7. **Repeat or quit**: Continue designing or enter `q` to quit

### Commands Between Generations

| Command | Description |
|---------|-------------|
| `c` | **Continue** - Parse population file and evaluate |
| `q` | **Quit** - Save progress and generate summary chart |
| `r` | **Refresh** - Show the menu again |

## Population JSON Format

The `population.json` file has this structure:

```json
{
  "_description": "...",
  "_available_terminals": ["GetRed", "GetGreen", "GetBlue", "GetGray"],
  "_available_primitives": { ... },
  "_parameter_ranges": { ... },
  "_previous_generation_results": [ ... ],
  "individuals": [
    "rf_classification(hog_features(GetRed), 100, 50)",
    "lr_classification(histogram_features(GetGray))",
    "sum_prediction_2(rf_classification(hog_features(GetRed), 200, 30), lr_classification(lbp_features(GetBlue)))"
  ]
}
```

**Only the `individuals` array matters** - other fields are for reference.

## S-Expression Syntax

S-expressions follow this pattern:
```
primitive(arg1, arg2, ...)
```

### Terminals (No Arguments)
- `GetRed` - Red channel of input image
- `GetGreen` - Green channel
- `GetBlue` - Blue channel  
- `GetGray` - Grayscale conversion

### Feature Extraction (Channel → FeatureVector)
```
histogram_features(GetRed)
hog_features(GetGray)
lbp_features(GetBlue)
sift_features(GetGreen)
sobel_features(GetRed)
gabor_features(GetGray, 0.785, 0.393)  # theta, frequency
gaussian_features(GetRed, 2)  # sigma
concat_images(GetRed, GetGreen)
```

### Filters (Channel → Channel)
```
mean_filter(GetRed)
gaussian_filter(GetGray, 2)  # sigma
sobel_filter(GetBlue)
laplacian_filter(GetRed)
median_filter(GetGreen)
```

### Classification (FeatureVector → Prediction)
```
rf_classification(features, 100, 50)   # trees, depth
erf_classification(features, 200, 30)  # trees, depth
lr_classification(features)
svm_classification(features)
```

### Feature Concatenation (FeatureVector × N → FeatureVector)
```
concat_features_2(feat1, feat2)
concat_features_3(feat1, feat2, feat3)
concat_features_4(feat1, feat2, feat3, feat4)
```

### Cascade Classification (FeatureVector → FeatureVector)
```
cascade_rf(features, 100, 50)
cascade_lr(features)
```

### Ensemble (Prediction × N → Prediction)
```
sum_prediction_2(pred1, pred2)
sum_prediction_3(pred1, pred2, pred3)
```

## Example Trees

### Simple: Single channel with HOG features
```
rf_classification(hog_features(GetRed), 100, 50)
```

### Moderate: Filtered channel with LBP features
```
lr_classification(lbp_features(gaussian_filter(GetGray, 2)))
```

### Complex: Multiple channels with feature concatenation
```
rf_classification(
  concat_features_2(
    hog_features(GetRed),
    lbp_features(GetBlue)
  ),
  200,
  50
)
```

### Ensemble: Multiple classifiers combined
```
sum_prediction_2(
  rf_classification(hog_features(GetRed), 100, 50),
  lr_classification(histogram_features(GetGray))
)
```

### Cascade: Stacked classifiers
```
rf_classification(
  cascade_lr(
    concat_features_2(
      hog_features(GetRed),
      sobel_features(GetGreen)
    )
  ),
  150,
  40
)
```

## Parameter Ranges

| Parameter | Range | Notes |
|-----------|-------|-------|
| Trees | 50-1000 | Step of 50 recommended |
| Depth | 10-100 | Step of 10 recommended |
| Sigma | 1-3 | Integer values |
| Theta | 0 to 7π/8 | Multiples of π/8 (use decimals: 0, 0.393, 0.785, 1.178, 1.571, 1.963, 2.356, 2.749) |
| Frequency | π/8 to π/2 | Multiples of π/8 (use decimals: 0.393, 0.785, 1.178, 1.571) |

## Error Handling

If your S-expressions have errors, the script will:
1. Show detailed error messages for each invalid expression
2. Tell you how many expressions parsed successfully
3. Allow you to fix the file and try again

Example error output:
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PARSING ERRORS DETECTED
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ✗ Individual 2: Unknown primitive or terminal: 'unknownFunc'
    Expression: unknownFunc(GetRed)...

----------------------------------------------------------------------
Successfully parsed: 3 individuals
Errors: 1

Please fix the errors in the JSON file and try again.
```

## Output Structure

```
experiments/
└── interactive_crc/
    ├── gen_0/
    │   ├── population.json       # Your input
    │   ├── generation_summary.json
    │   ├── tree_0/
    │   │   ├── results.json
    │   │   └── tree_viz.png
    │   ├── tree_1/
    │   │   └── ...
    │   └── ...
    ├── gen_1/
    │   ├── population.json       # Template with previous results
    │   └── ...
    ├── experiment_summary.json
    └── progress_plot.png
```

## Programmatic Usage

```python
from src import gp_setup
from src.gp_experiment_interactive import run_interactive_experiment

# Load your data
X_train, y_train = ...
X_val, y_val = ...
X_test, y_test = ...

# Create primitive set
pset = gp_setup.create_primitive_set()

# Run interactive experiment
runner = run_interactive_experiment(
    experiment_name="my_interactive_exp",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    pset=pset,
    batch_size=32,
    n_workers=4,  # or None for auto-detect
    max_generations=10,  # or None for unlimited
    output_dir="experiments"
)
```

## Tips

1. **Start simple**: Begin with basic trees to understand the primitives
2. **Check results**: Previous generation results are included in the template
3. **Iterate**: Build on successful trees from previous generations
4. **Combine**: Use ensemble primitives to combine multiple approaches
5. **Filter first**: Image filters can improve feature extraction
