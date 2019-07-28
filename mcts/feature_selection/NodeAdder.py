class NodeAdder:
    def __init__(self, root):
        self._nodes_buckets = {}
        self._nodes_buckets[0] = [root]
        
    def add_node(self, node, feature_name):
        """
        Method for adding node to tree
        Parameters
        ----------
        node: Node
            Node that is parent of new node
        feature_name: str
            Name of new feature in path
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