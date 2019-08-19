import numpy as np
from gsfs.feature_selection.LRavePaths import *

class Node:
    """Class representing node of MCTS tree."""
    def __init__(self, used_features, feature_name):
        self._scores = []
        self._scores_sum = 0
        self.T = 0
        features = used_features.copy()
        if feature_name is not None:
            features.add(feature_name)
        self._features = features
        self._children = []
    
    def add_child(self, node):
        """
        Method for adding child node
        Parameters
        ----------
        node: Node
            Node that will be added as child
        """
        
        self._children.append(node)
    
    def add_score(self, score):
        """
        Method for adding score to node
        Parameters
        ----------
        score: float
            Value of the score that will be added
        """
        
        self._scores.append(score)
        self.T += 1
        self._scores_sum += score

    def get_variance(self):
        """Method for getting variance of scores of node, if node not visited then 0 is returned"""
        return np.var(self._scores) if self.T != 0 else 0
    
    def get_score(self):
        """Method for getting score for node, if node not visited then 0 is returned"""
        return self._scores_sum/self.T if self.T != 0 else 0
        
    def get_label(self):
        """Method for getting label for node"""
        
        return ','.join(sorted(self._features))
        
    def get_str_node_info(self):
        """
        Method for getting string label for node
        """
        
        return '''T: {:d}
        avg score: {:.4f}
        var: {:.4f}'''.format(self.T, (self._scores_sum/self.T if self.T != 0 else 0), (np.var(self._scores) if self.T != 0 else 0))
    
    def get_used_features_in_children(self):
        used_features = set()
        used_features.update(self._features)
        
        for node in self._children:
            used_features.update(node._features)
            
        return used_features