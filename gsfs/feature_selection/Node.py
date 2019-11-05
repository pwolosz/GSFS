import numpy as np
from gsfs.feature_selection.LRavePaths import *

class Node:
    """Class representing of the search graph’s node. 
    Every node is keeping all scores that were added to this node, 
    this list of scores is used to calculate the average score for the node and variance of scores, 
    both these values are used in calculating the score of the node during the searching of the graph."""

    def __init__(self, used_features, feature_name):
        """
        used_features: set
            Set of features that are used in the node to which the new node will be connected to first,
        feature_name: str
            Name of the feature that will be added to used_features and the newly created set will be the set of features that this node is representing.
        """

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
        Method for adding new child node to current node.

        Parameters
        ----------
        node: gsfs.feature_selection.Node
            Node that will be added as child to current node.

        Returns: None
        """
        
        self._children.append(node)
    
    def add_score(self, score):
        """
        Method for adding score to current node. Used after every search iteration to propagate the scores to all nodes in a searched path of the graph.

        Parameters
        ---------- 
        score: float
            Score that will be added

        Returns: None
        """
        
        self._scores.append(score)
        self.T += 1
        self._scores_sum += score

    def get_variance(self):
        """
        Method for getting variance of scores for current node, if node hasn’t been visited yet (has no scores) then 0 is returned.

        Returns: float
            Variance of the node.
        """
        
        return np.var(self._scores) if self.T != 0 else 0
    
    def get_score(self):
        """
        Method for getting average score for current node, if node hasn’t been visited yet (has no scores) then 0 is returned.

        Returns: float
            average score for node.
        """

        return self._scores_sum/self.T if self.T != 0 else 0
        
    def get_label(self):
        """
        Method for getting concatenated features of current node, separated by commas.

        Returns: str
            Label of the node, e.g. "feature1,feature2".
        """
        
        return ','.join(sorted(self._features))
        
    def get_str_node_info(self):
        """
        Method for current node’s information as a string. This information is average score, number of visits and variance of scores.

        Returns: str
            Multi-line string with informations about node.
        """
        
        return '''T: {:d}
        avg score: {:.4f}
        var: {:.4f}'''.format(self.T, (self._scores_sum/self.T if self.T != 0 else 0), (np.var(self._scores) if self.T != 0 else 0))
    
    def get_used_features_in_children(self):
        """
        Method for getting set of used features current node’s children nodes.

        Returns: set
            Set of features that are used in node’s children nodes
        """

        used_features = set()
        used_features.update(self._features)
        
        for node in self._children:
            used_features.update(node._features)
            
        return used_features