from graphviz import Digraph
from queue import *

def draw_tree(node_adder, file_name = None, view = True, view_nodes_info = False):
    """
    Method for drawing tree build during MCTS
    Parameters
    ----------
    node_adder: NodeAdder
        NodeAdder used in mcts iterations
    file_name: str
        Name of the output file, if None then the visualization won't be saved
    view: boolean
        Flag indicating whether the visualization will be shown
    view_nodes_info: boolean
        
    """
    
    dot = Digraph(comment='mcts')

    # Adding nodes
    for key, value in node_adder._nodes_buckets.items():
        for node in value:
            label = node.get_label()
            dot.node(label,label)
    
    # Adding edges
    for key, value in node_adder._nodes_buckets.items():
        for node in value:
            for child_node in node._children:
                dot.edge(node.get_label(), child_node.get_label())
            
    dot.render(file_name, view = view)