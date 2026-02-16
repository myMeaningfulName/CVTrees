import networkx as nx
import matplotlib.pyplot as plt
from deap import gp

def draw_tree(individual, filename="tree.png"):
    """
    Draws the GP tree using NetworkX and Matplotlib.
    Ensures evenly spaced charts regardless of tree size.
    """
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # Create a layout
    pos = _hierarchy_pos(g, 0)

    # Dynamically adjust figure size based on tree depth and width
    max_depth = max([pos[node][1] for node in pos]) if pos else 1
    max_width = len(pos) if pos else 1
    fig_width = max(12, max_width * 1.5)
    fig_height = max(8, max_depth * 2)

    plt.figure(figsize=(fig_width, fig_height))
    nx.draw_networkx_nodes(g, pos, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(g, pos, labels, font_size=10, font_weight="bold")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Tree saved to {filename}")
    plt.close()

def _hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    """
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    """
    if not nx.is_tree(G):
        # Fallback for non-trees (should not happen in GP)
        return nx.spring_layout(G)

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
