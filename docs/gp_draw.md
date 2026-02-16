# GP Draw Module (`gp_draw.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [The `draw_tree` Function](#the-draw_tree-function)
4. [The Hierarchical Layout Algorithm](#the-hierarchical-layout-algorithm)
5. [Customization Options](#customization-options)
6. [Step-by-Step Visualization Example](#step-by-step-visualization-example)
7. [Complete Code Walkthrough](#complete-code-walkthrough)

---

## Overview

The `gp_draw.py` module provides visualization capabilities for GP trees. It converts DEAP's tree representation into a visual graph using NetworkX and Matplotlib.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. DEAP Tree (List representation)                             │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ [rf_classification, hog_features, GetRed, 100, 50]        │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│   2. DEAP's gp.graph()                                           │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ nodes = [0, 1, 2, 3, 4]                                   │  │
│   │ edges = [(0,1), (0,3), (0,4), (1,2)]                      │  │
│   │ labels = {0:'rf_classification', 1:'hog_features', ...}   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│   3. NetworkX Graph                                              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ G.nodes = [0, 1, 2, 3, 4]                                 │  │
│   │ G.edges = [(0,1), (0,3), (0,4), (1,2)]                    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│   4. Hierarchical Layout                                         │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ pos = {0:(0.5, 0), 1:(0.25, -0.2), 2:(0.25, -0.4), ...}   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│   5. Matplotlib Rendering                                        │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ 📊 Visual tree saved to tree_viz.png                      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```python
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp
```

### NetworkX

[NetworkX](https://networkx.org/) is a Python library for creating, manipulating, and analyzing graphs and networks.

**Key concepts used**:
- `nx.Graph()`: Creates an undirected graph
- `G.add_nodes_from(nodes)`: Add multiple nodes
- `G.add_edges_from(edges)`: Add multiple edges
- `nx.is_tree(G)`: Check if graph is a tree
- `nx.draw_networkx_*()`: Drawing functions

### Matplotlib

[Matplotlib](https://matplotlib.org/) is the standard Python plotting library.

**Key concepts used**:
- `plt.figure(figsize=...)`: Create a figure with specific size
- `plt.savefig()`: Save figure to file
- `plt.close()`: Close figure (free memory)

### DEAP's `gp.graph()`

DEAP provides a utility to convert a tree to graph components:

```python
from deap import gp

tree = [rf_classification, hog_features, GetRed, 100, 50]
nodes, edges, labels = gp.graph(tree)

# nodes: [0, 1, 2, 3, 4]  (node indices)
# edges: [(0, 1), (0, 3), (0, 4), (1, 2)]  (parent-child pairs)
# labels: {0: 'rf_classification', 1: 'hog_features', 2: 'GetRed', ...}
```

---

## The `draw_tree` Function

### Signature

```python
def draw_tree(individual, filename="tree.png"):
    """
    Draws the GP tree using NetworkX and Matplotlib.
    Ensures evenly spaced charts regardless of tree size.
    """
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `individual` | `gp.PrimitiveTree` | Required | The tree to visualize |
| `filename` | `str` | `"tree.png"` | Output file path |

### What It Does

1. **Convert tree to graph components** using DEAP's `gp.graph()`
2. **Create NetworkX graph** from nodes and edges
3. **Compute hierarchical layout** positions
4. **Determine figure size** based on tree dimensions
5. **Draw nodes, edges, and labels**
6. **Save to file**

### Full Implementation

```python
def draw_tree(individual, filename="tree.png"):
    # Step 1: Get graph components from DEAP
    nodes, edges, labels = gp.graph(individual)
    
    # Step 2: Create NetworkX graph
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # Step 3: Compute layout
    pos = _hierarchy_pos(g, 0)  # 0 is root

    # Step 4: Determine figure size
    max_depth = max([pos[node][1] for node in pos]) if pos else 1
    max_width = len(pos) if pos else 1
    fig_width = max(12, max_width * 1.5)
    fig_height = max(8, max_depth * 2)

    # Step 5: Draw
    plt.figure(figsize=(fig_width, fig_height))
    nx.draw_networkx_nodes(g, pos, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(g, pos, labels, font_size=10, font_weight="bold")

    # Step 6: Save
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Tree saved to {filename}")
    plt.close()
```

### Step-by-Step Breakdown

#### Step 1: Get Graph Components

```python
nodes, edges, labels = gp.graph(individual)
```

DEAP's `gp.graph()` traverses the tree and extracts:
- **nodes**: List of integers `[0, 1, 2, ...]` (one per tree node)
- **edges**: List of tuples `[(0, 1), (0, 2), ...]` (parent-child relationships)
- **labels**: Dictionary `{0: 'function_name', 1: 'terminal_name', ...}`

#### Step 2: Create NetworkX Graph

```python
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
```

Creates an undirected graph. Note: We use `nx.Graph()` (undirected) rather than `nx.DiGraph()` (directed) because the layout algorithm works with undirected graphs.

#### Step 3: Compute Layout

```python
pos = _hierarchy_pos(g, 0)
```

Calls the custom layout function (explained below) starting from node 0 (the root).

Returns a dictionary: `{node_id: (x, y), ...}`

#### Step 4: Dynamic Figure Size

```python
max_depth = max([pos[node][1] for node in pos]) if pos else 1
max_width = len(pos) if pos else 1
fig_width = max(12, max_width * 1.5)
fig_height = max(8, max_depth * 2)
```

**Why dynamic sizing?**
- Small trees (3-5 nodes) would look cramped in a huge figure
- Large trees (50+ nodes) would be unreadable in a small figure

**Formulas**:
- `fig_width`: At least 12 inches, grows with number of nodes
- `fig_height`: At least 8 inches, grows with tree depth

The `max_depth` uses negative y-values (depth goes down), so we need the max of the y-coordinates.

#### Step 5: Draw Components

```python
plt.figure(figsize=(fig_width, fig_height))

# Draw nodes as circles
nx.draw_networkx_nodes(g, pos, node_size=1000, node_color="lightblue")

# Draw edges as lines
nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)

# Draw labels inside nodes
nx.draw_networkx_labels(g, pos, labels, font_size=10, font_weight="bold")
```

**Parameters explained**:
- `node_size=1000`: Size in points squared (fairly large)
- `node_color="lightblue"`: Fill color
- `width=1.0`: Edge line width
- `alpha=0.5`: Edge transparency (50%)
- `font_size=10`: Label text size
- `font_weight="bold"`: Make labels stand out

#### Step 6: Save and Cleanup

```python
plt.axis("off")      # Hide axis lines and ticks
plt.tight_layout()   # Reduce whitespace
plt.savefig(filename)
print(f"Tree saved to {filename}")
plt.close()          # Free memory
```

---

## The Hierarchical Layout Algorithm

### Purpose

Trees should be drawn with:
- **Root at top**
- **Children below parents**
- **Siblings spread horizontally**
- **Even spacing**

NetworkX's built-in layouts (spring, circular, etc.) don't enforce this hierarchy.

### The `_hierarchy_pos` Function

```python
def _hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    """
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `G` | Required | NetworkX graph |
| `root` | `None` | Root node ID (usually 0) |
| `width` | `1.0` | Total horizontal space available |
| `vert_gap` | `0.2` | Vertical distance between levels |
| `vert_loc` | `0` | Y-coordinate of root |
| `xcenter` | `0.5` | X-coordinate of root (centered) |

### Fallback for Non-Trees

```python
if not nx.is_tree(G):
    return nx.spring_layout(G)
```

If the graph isn't a tree (shouldn't happen with GP), fall back to spring layout.

### The Recursive Algorithm

```python
def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, 
                   xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  # Don't go back up!
    
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                pos=pos, parent=root)
    return pos
```

### Algorithm Walkthrough

**Initial call**:
```python
_hierarchy_pos(G, root=0, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5)
```

**Step 1**: Position root at (0.5, 0)

**Step 2**: Get children
- `G.neighbors(0)` returns connected nodes
- Remove `parent` to avoid going back up

**Step 3**: Divide horizontal space among children
- If 2 children: each gets width/2 = 0.5
- Position at: 0.25 and 0.75

**Step 4**: Recurse for each child
- Move down: `vert_loc - vert_gap`
- Narrow width: `dx` (child's portion)
- New center: computed position

### Visual Example

```
Tree: rf_classification(hog_features(GetRed), 100, 50)

Level 0:                rf_classification (0.5, 0)
                       /         |         \
Level 1:    hog_features      100         50
            (0.17, -0.2)  (0.5, -0.2) (0.83, -0.2)
                |
Level 2:     GetRed
            (0.17, -0.4)
```

**Coordinate calculation for level 1**:
- 3 children, width = 1.0
- `dx = 1.0 / 3 = 0.333`
- `nextx` starts at `0.5 - 0.5 - 0.167 = -0.167`
- Child 1: `nextx += 0.333` → 0.167
- Child 2: `nextx += 0.333` → 0.5
- Child 3: `nextx += 0.333` → 0.833

---

## Customization Options

### Changing Node Appearance

```python
# Different colors
nx.draw_networkx_nodes(g, pos, node_size=1000, 
                       node_color=["red" if "classification" in labels.get(n, "") 
                                   else "lightblue" for n in g.nodes()])

# Different sizes
sizes = [2000 if labels.get(n, "").startswith("Get") else 1000 for n in g.nodes()]
nx.draw_networkx_nodes(g, pos, node_size=sizes, node_color="lightblue")
```

### Changing Edge Appearance

```python
# Thicker, darker edges
nx.draw_networkx_edges(g, pos, width=2.0, alpha=0.8, edge_color="gray")

# Arrows (for directed graphs)
nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=20)
```

### Changing Labels

```python
# Shorter labels
short_labels = {k: v[:8] + "..." if len(v) > 10 else v 
                for k, v in labels.items()}
nx.draw_networkx_labels(g, pos, short_labels, font_size=8)

# Labels outside nodes
label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
nx.draw_networkx_labels(g, label_pos, labels, font_size=8)
```

### Changing Layout Parameters

```python
# More vertical spacing
pos = _hierarchy_pos(g, 0, vert_gap=0.4)

# Narrower tree
pos = _hierarchy_pos(g, 0, width=0.5)
```

---

## Step-by-Step Visualization Example

### Example Tree

```python
# Tree expression:
# sum_prediction_2(
#     rf_classification(hog_features(GetRed), 100, 50),
#     lr_classification(lbp_features(GetGray))
# )

# DEAP representation (prefix notation):
tree = [sum_prediction_2, rf_classification, hog_features, GetRed, 100, 50,
        lr_classification, lbp_features, GetGray]
```

### Step 1: Extract Graph Components

```python
nodes, edges, labels = gp.graph(tree)

# nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# edges = [(0, 1), (0, 6), (1, 2), (1, 4), (1, 5), (2, 3), (6, 7), (7, 8)]
# labels = {
#     0: 'sum_prediction_2',
#     1: 'rf_classification',
#     2: 'hog_features',
#     3: 'GetRed',
#     4: '100',
#     5: '50',
#     6: 'lr_classification',
#     7: 'lbp_features',
#     8: 'GetGray'
# }
```

### Step 2: Create Graph

```python
g = nx.Graph()
g.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
g.add_edges_from([(0, 1), (0, 6), (1, 2), (1, 4), (1, 5), (2, 3), (6, 7), (7, 8)])
```

### Step 3: Compute Layout

```
Level 0:                    sum_prediction_2 (0.5, 0)
                           /                 \
Level 1:       rf_classification        lr_classification
                 (0.25, -0.2)             (0.75, -0.2)
              /      |      \                   |
Level 2: hog_features  100   50          lbp_features
         (0.08, -0.4)  (0.25) (0.42)      (0.75, -0.4)
              |                                 |
Level 3:   GetRed                           GetGray
         (0.08, -0.6)                     (0.75, -0.6)
```

### Step 4: Render

The final visualization would look like:

```
                    ┌───────────────────┐
                    │ sum_prediction_2  │
                    └─────────┬─────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                                   │
    ┌───────┴───────┐                   ┌───────┴───────┐
    │rf_classification│                   │lr_classification│
    └───────┬───────┘                   └───────┬───────┘
            │                                   │
    ┌───────┼───────┐                   ┌───────┴───────┐
    │       │       │                   │ lbp_features  │
┌───┴───┐ ┌─┴─┐ ┌─┴─┐                   └───────┬───────┘
│hog_   │ │100│ │50 │                           │
│features│ └───┘ └───┘                   ┌───────┴───────┐
└───┬───┘                               │   GetGray     │
    │                                   └───────────────┘
┌───┴───┐
│GetRed │
└───────┘
```

---

## Complete Code Walkthrough

```python
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp
```

**Lines 1-3**: Import graph library, plotting library, and DEAP's GP module.

---

```python
def draw_tree(individual, filename="tree.png"):
    """
    Draws the GP tree using NetworkX and Matplotlib.
    Ensures evenly spaced charts regardless of tree size.
    """
```

**Lines 5-9**: Function definition with docstring.

---

```python
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
```

**Lines 10-13**: Convert DEAP tree to NetworkX graph.

---

```python
    pos = _hierarchy_pos(g, 0)
```

**Line 16**: Compute node positions using hierarchical layout.

---

```python
    max_depth = max([pos[node][1] for node in pos]) if pos else 1
    max_width = len(pos) if pos else 1
    fig_width = max(12, max_width * 1.5)
    fig_height = max(8, max_depth * 2)
```

**Lines 19-22**: Calculate figure dimensions based on tree size.

Note: `max_depth` will be a negative number (y goes down), so we're finding the "most negative" which is the deepest level. Then we multiply by 2 for height (and since it's negative, this works out to give positive height).

Actually, looking at the code more carefully, `max_depth * 2` with a negative value would give a negative result, which is why we use `max(8, ...)` - it ensures at least 8 inches. The formula could be improved to use `abs()`.

---

```python
    plt.figure(figsize=(fig_width, fig_height))
    nx.draw_networkx_nodes(g, pos, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(g, pos, labels, font_size=10, font_weight="bold")
```

**Lines 24-27**: Create figure and draw graph components.

---

```python
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Tree saved to {filename}")
    plt.close()
```

**Lines 29-33**: Finalize and save the figure.

---

```python
def _hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    """
    if not nx.is_tree(G):
        return nx.spring_layout(G)
```

**Lines 35-42**: Outer function definition with fallback for non-trees.

---

```python
    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, 
                       xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
```

**Lines 44-49**: Inner recursive function. Initialize or update position dictionary.

---

```python
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
```

**Lines 50-52**: Get children of current node. For undirected graphs, neighbors include the parent, so we remove it.

---

```python
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                    pos=pos, parent=root)
        return pos
```

**Lines 53-61**: Recursively position children. Each child gets an equal portion of the horizontal space, positioned one level down.

---

```python
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
```

**Line 63**: Start the recursion from the root.

---

## Summary

The `gp_draw.py` module provides simple but effective tree visualization:

| Function | Purpose |
|----------|---------|
| `draw_tree` | Main entry point - draws and saves tree image |
| `_hierarchy_pos` | Computes node positions for hierarchical layout |

Key features:
- Converts DEAP trees to visual graphs
- Uses hierarchical layout (root at top)
- Dynamically sizes figures based on tree dimensions
- Saves to PNG files

The visualizations help understand evolved trees and are saved for each individual in the experiment.
