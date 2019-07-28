import math

class MultiArmStrategies:
    """Class for representing strategies used for selecting next node to visit in MCTS. 
    For more info please see documentation"""
    
    def __init__(self, name, all_node_names, params):
        """
        Parameters
        ----------
        name: str
            Name of the strategy that will be used
        all_node_names: list
            List containing all possible names of the features that MCTS will search through
        """
        
        self._name = name
        self._all_features_names = all_features_names
        self._params = params
        
    def multiarm_strategy(self, node, scoring_functions, global_scores, node_adder):
        """
        Method for getting next node in current search.
        Parameters
        ----------
        node: Node
            Current node
        used_features: list
            List of feature names that are already used in search path
        scoring_functions: ScoringFunctions
            Object containing functions for calculating scores
        global_scores: GlobalScores
            Object containing scores used in strategy
        params: dict
            Dictionary with MCTS parameters
        """
        
        if self._name == 'continuous':
            return self._continuous_strategy(node, scoring_functions, global_scores)
        elif self._name == 'discrete':
            return self._discrete_strategy(node, scoring_functions, global_scores, node_adder)
        else:
            raise Exception("Error getting multiarm strategy, strategy \'" + self._name + "\' is not supported.")   
    
    def _continuous_strategy(self, node, scoring_functions, global_scores):
        if(len(node.child_nodes) == 0):
            self._add_all_child_nodes(node, used_features)
            return node.child_nodes[0]
        else:
            return self._get_best_node(node, scoring_functions, global_scores)

    def _discrete_strategy(self, node, scoring_functions, global_scores, node_adder):
        if len(node.child_nodes) == 0 or self._should_add_child(node):
            return self._add_child_node(node, scoring_functions, global_scores, node_adder)
        else:
            return self._get_best_node(node, scoring_functions, global_scores)
                
        
    def _should_add_child(self, node):
        return (((int(math.pow(node.T, self._params['b_T'])) - int(math.pow(node.T - 1, self._params['b_T']))) > 0) and
               not self._all_features_names.issubset(node._features))
    
    def _get_best_node(self, node, scoring_functions, global_scores):
        best_score = 0
        best_node = None
        tmp_score = 0
            
        for child_node in node._children:
            score = scoring_functions.get_score(node, child_node, global_scores)
            if score > best_score:
                best_score = score
                best_node = child_node
                
        return best_node
    
    def _add_child_node(self, node, scoring_functions, global_scores, node_adder):
        best_score = 0
        best_feature = None
        not_used_features = self._all_features_names - node._features
        
        for feature in not_used_features:
            score = scoring_functions.get_new_node_score(feature, node._features, global_scores)
            if score > best_score:
                best_feature = feature
                best_score = score
        
        if best_feature is not None:
            return node_adder.add_node(node._features, best_feature)
            
        return node_adder.add_node(node._features, not_used_features.pop())
        
    def _add_all_child_nodes(self, node, used_features):
        node.add_child_nodes(self._all_node_names - used_features)   