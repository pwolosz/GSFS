import numpy as np

class Node:
    """Class representing node of MCTS tree."""
    def __init__(self, feature_name, parent_node = None, is_subtree_full = False):
        """
        Parameters
        ----------
        feature_name: str
            Name of the feature that this node will represent
        parent_node: Node (default: None)
            Reference to parent node of new node, if None that means that it's root
        is_subtree_full: boolean (default: False)
            Value indicating whether the node's subtree is fully searched, if so then this node won't be searched in the
            future. If parameter set to True then it's last node in the path from the root (all features used in this path)
        """
        
        self.feature_name = feature_name
        self.child_nodes = []
        self.T = 0
        self.score_sum = 0
        self._is_subtree_full = is_subtree_full
        self._parent_node = parent_node
        self._scores = []
        self.id = str(id(self))
        
    def add_child_node(self, node_name, is_subtree_full = False):
        """
        Method for adding child node to children of current node.
        Parameters
        ----------
        node_name: str
            Name (feature name) of the node that will be added
        is_subtree_full: boolean (default: False)
            Value indicating whether the node will be last feature not used in path from root to current node
        """
        
        new_node = Node(node_name, self, is_subtree_full)
        self.child_nodes.append(new_node)
        
        return new_node
        
    def add_child_nodes(self, node_names):
        """
        Method for adding child nodes to children of current node.
        Parameters
        ----------
        node_names: list
            List of names (feature names) of nodes that will be added to children of current node
        """
        
        for name in node_names:
            self.add_child_node(name, len(node_names) == 1)
      
    def update_node(self, score, scores):
        """
        Method for updating scores of current node.
        Parameters
        ----------
        score: numeric
            Value of the score that will added to score of the current node
        scores: GlobalScores
            Instance of GlobalScores
        """
        
        self.score_sum += score
        self.T += 1
        self._scores.append(score)
        scores.update_g_rave_score(self.feature_name, score)
        if(len(self.child_nodes) != 0):
            is_subtree_full = True
            for node in self.child_nodes:
                if(not node._is_subtree_full):
                    is_subtree_full = False
                    break
            self._is_subtree_full = is_subtree_full
    
    def get_score(self):
        """Method for getting score for current node, if node not visited then float(\'Inf\') is returned, 
        otherwise numeric is returned"""
        
        return float('Inf') if self.T == 0 else self.score_sum/self.T
    
    def get_variance(self):
        """Method for getting variance of scores of current node."""
        
        return np.var(self._scores)
    
    def update_scores_up(self, score, scores):
        """
        Method for updating scores of current node and updating scores of all nodes up to the root
        Parameters
        ----------
        score: numeric
            Value of the score that will added to score of the current node
        scores: GlobalScores
            Instance of GlobalScores
        """
        
        current_node = self
        
        while(current_node != None):
            current_node.update_node(score, scores)
            current_node = current_node._parent_node  
            
    def get_str_node_info(self):
        """
        Method for getting string label for node
        """
        
        return '''T: {:d}
        avg score: {:.4f}
        var: {:.4f}
        '''.format(self.T, self.score_sum/self.T if self.T != 0 else 0, np.var(self._scores) if self.T != 0 else 0)