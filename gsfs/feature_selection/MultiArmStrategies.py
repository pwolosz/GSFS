import math
import random

class MultiArmStrategies:
    """Class containing functions for multi-arm strategies that are used during search of the graph."""
    
    def __init__(self, name, all_features_names, params):
        """
        Parameters
        ----------
        name: str
            Name of the strategy, available values are "discrete" and "continuous",
        all_features_names: set
            Set containing all names of the variables in adataset used in search,
        params: dict
            Parameters of the algorithm.
        """
        
        self._name = name
        self._all_features_names = all_features_names
        self._params = params
        
    def multiarm_strategy(self, node, scoring_functions, global_scores, node_adder):
        """
        Method for getting next node in a search using strategy specified in constructor. 
        If needed,the method adds a new node to the graph.

        Parameters
        ----------
        node: gsfs.feature_selection.Node
            Current node in a search iteration,
        scoring_functions: gsfs.feature_selection.ScoringFunctions
            Scoring-Functions object used in a search, containing methods for calculating scoresfor nodes,
        global_scores: gsfs.feature_selection.GlobalScores
            GlobalScores ob-ject used in a search, contains RAVE scores,
        node_adder: gsfs.feature_selection.NodeAdder
            Object used in search, allows MultiArmStrategies to add nodes to current node.
            
        Returns: gsfs.feature_selection.Node
            Next node in search.
        """
        
        if self._name == 'continuous':
            return self._continuous_strategy(node, scoring_functions, global_scores, node_adder)
        elif self._name == 'discrete':
            return self._discrete_strategy(node, scoring_functions, global_scores, node_adder)
        else:
            raise Exception("Error getting multiarm strategy, strategy \'" + self._name + "\' is not supported.")   
    
    def _continuous_strategy(self, node, scoring_functions, global_scores, node_adder):
        return self._get_best_node_continuous(node, scoring_functions, global_scores, node_adder)

    def _discrete_strategy(self, node, scoring_functions, global_scores, node_adder):
        if self._should_add_child(node):
            return self._add_child_node(node, scoring_functions, global_scores, node_adder)
            
        return self._get_best_node(node, scoring_functions, global_scores)
                
        
    def _should_add_child(self, node):
        if node.T == 0:
            return True
        return (((int(math.pow(node.T, self._params['b_T'])) - int(math.pow(node.T - 1, self._params['b_T']))) > 0) and
               not self._all_features_names.issubset(node._features))
    
    def _get_best_node(self, node, scoring_functions, global_scores):
        best_score = -1
        best_node = None

        for child_node in node._children:
            score = scoring_functions.get_score(node, child_node, global_scores)
            
            if score > best_score:
                best_score = score
                best_node = child_node
       
        return best_node
    
    def _get_best_node_continuous(self, node, scoring_functions, global_scores, node_adder):
        best_score = -1
        best_node = None
        not_used_features = self._all_features_names - node._features - node.get_used_features_in_children()
        add_node = False
        best_feature = None
        for child_node in node._children:
            score = scoring_functions.get_score(node, child_node, global_scores)
           
            if score > best_score:
                best_score = score
                best_node = child_node    
        for feature in not_used_features:
            score = scoring_functions.get_new_node_score(feature, node, global_scores)
            
            if score * self._params['new_node_preference'] > best_score:
                best_score = score
                add_node = True
                best_feature = feature
            
        if add_node:
            best_node = node_adder.add_node(node, best_feature)
            
        return best_node
    
    def _add_child_node(self, node, scoring_functions, global_scores, node_adder):
        best_score = 0
        best_feature = None
        not_used_features = self._all_features_names - node._features - node.get_used_features_in_children()
        
        if len(not_used_features) == 0:
            return self._get_best_node(node, scoring_functions, global_scores)
        
        for feature in not_used_features:
            score = scoring_functions.get_new_node_score(feature, node, global_scores)
            if score > best_score:
                best_feature = feature
                best_score = score
        
        if best_feature is not None:
            return node_adder.add_node(node, best_feature)
        
        return node_adder.add_node(node, list(not_used_features)[random.randint(0,len(not_used_features))])
        
    def _add_all_child_nodes(self, node, used_features):
        node.add_child_nodes(self._all_node_names - used_features)   
        