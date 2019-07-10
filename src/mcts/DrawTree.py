from graphviz import Digraph
from queue import *

def draw_tree(root, file_name = None, view = True, view_nodes_info = False):
    """
    Method for drawing tree build during MCTS
    Parameters
    ----------
    root: Node
        Root of the tree
    file_name: str
        Name of the output file, if None then the visualization won't be saved
    view: boolean
        Flag indicating whether the visualization will be shown
    view_nodes_info: boolean
        
    """
    
    q = Queue()
    dot = Digraph(comment='mcts')

    q.put(root)
    dot.node(str(id(root)), root.feature_name + '\n' + root.get_str_node_info() if view_nodes_info else root.feature_name)

    while not q.empty():
        curr_node = q.get()
        edges = []
        for node in curr_node.child_nodes:
            edges.append(str(id(curr_node))+str(id(node)))
            node_label = node.feature_name
            
            if view_nodes_info:
                node_label += '\n' + node.get_str_node_info()
            
            dot.node(str(id(node)), node_label)
            dot.edge(str(id(curr_node)), str(id(node)))
            q.put(node)
    
    dot.render(file_name, view = view)