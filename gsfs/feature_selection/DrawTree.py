from graphviz import Digraph
from queue import *

def draw_tree(node_adder, file_name = None, view = True, view_nodes_info = False):
    """
    Method for drawing the search graph.

    Parameters
    ----------
    node_adder: gsfs.feature_selection.NodeAdder
        NodeAdder instance that was used in algorithm,
    file_name: str (default: None)
        If specified the graph will be saved in pdf format in a file specified in this parameter,
    view: boolean (default: True)
        If Truethen the graph will beviewed in a default browser,
    view_nodes_info: boolean (default: False)
        If False then only features names are displayed, otherwise each node will have additionaly number of visits, 
        average score and variance displayed.
       
    Returns: None
    """
    
    dot = Digraph(comment='mcts')

    # Adding nodes
    for key, value in node_adder._nodes_buckets.items():
        for node in value:
            node_id = node.get_label()
            label = node_id

            if view_nodes_info:
                label += '\n' + node.get_str_node_info()
            
            dot.node(node_id,label)
    
    # Adding edges
    for key, value in node_adder._nodes_buckets.items():
        for node in value:
            for child_node in node._children:
                
                dot.edge(node.get_label(), child_node.get_label())
            
    dot.render(file_name, view = view)