from gsfs.feature_selection.Node import *

class NodeAdder:
    """Class that is used to add nodes to algorithm’s search graph."""

    def __init__(self, root):
        """
        root: gsfs.feature_selection.Node
        Root of the search graph used in algorithm.
        """
        
        self._nodes_buckets = {}
        self._nodes_buckets[0] = [root]
        
    def add_node(self, node, feature_name):
        """
        method that adds new node to selected node.
        Parameters
        ----------
        node: gsfs.feature_selection.Node
            Parent node of the newly added node,
        feature_name: str
            New feature, new node's features = node.features + [feature_name].

        Returns: gsfs.feature_selection.Node
            newly added node.
        """

        new_node = Node(node._features, feature_name)    
        
        if len(node._features) + 1 not in self._nodes_buckets:
            self._nodes_buckets[len(node._features) + 1] = []
            
        self._nodes_buckets[len(node._features) + 1].append(new_node)
        
        for prev_node in self._nodes_buckets[len(node._features)]:
            if prev_node._features.issubset(new_node._features):
                prev_node.add_child(new_node)
                
        if len(node._features) + 2 in self._nodes_buckets:
            for next_node in self._nodes_buckets[len(node._features) + 2]:
                if new_node._features.issubset(next_node._features):
                    new_node.add_child(next_node)
                    
        return new_node