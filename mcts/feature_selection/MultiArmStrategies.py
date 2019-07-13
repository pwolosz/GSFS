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
        self._all_node_names = all_node_names
        self._params = params
        
    def multiarm_strategy(self, node, used_features, scoring_functions, global_scores, params):
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
            return self._continuous_strategy(node, used_features, scoring_functions, global_scores)
        elif self._name == 'discrete':
            return self._discrete_strategy(node, used_features, scoring_functions, global_scores)
        else:
            raise Exception("Error getting multiarm strategy, strategy \'" + self._name + "\' is not supported.")   
    
    def _continuous_strategy(self, node, used_features, scoring_functions, global_scores):
        if(len(node.child_nodes) == 0):
            self._add_all_child_nodes(node, used_features)
            return node.child_nodes[0]
        else:
            return self._get_best_node(node, scoring_functions, global_scores)

    def _discrete_strategy(self, node, used_features, scoring_functions, global_scores):
        if (len(node.child_nodes) == 0 or self._should_add_child(node)) and not node.all_children_added:
            return self._add_child_node(node, scoring_functions, global_scores, used_features)
        else:
            return self._get_best_node(node, scoring_functions, global_scores)
                
        
    def _should_add_child(self, node):
        return (int(math.pow(node.T, self._params['b_T'])) - int(math.pow(node.T - 1, self._params['b_T']))) > 0
    
    def _get_best_node(self, node, scoring_functions, global_scores):
        best_score = 0
        best_node = None
        tmp_score = 0
            
        for child_node in node.child_nodes:
            if(not child_node._is_subtree_full):
                score = scoring_functions.get_score(child_node, global_scores)
                if(score > best_score):
                    best_score = score
                    best_node = child_node
        return best_node
    
    def _add_child_node(self, node, scoring_functions, global_scores, used_features):
        best_score = 0
        best_feature = None
        not_used_features = self._all_node_names - used_features
        
        for tmp_node in node.child_nodes:
            not_used_features.remove(tmp_node.feature_name)
        
        
        for feature in not_used_features:
            curr_score = scoring_functions.get_feature_score(feature, global_scores)
            if curr_score > best_score:
                best_score = curr_score
                best_feature = feature
  

        if len(not_used_features) == 1:
            node.all_children_added = True
        
        if best_feature is not None:
            return node.add_child_node(best_feature)
        else:
            return node.add_child_node((self._all_node_names - used_features)[0])
        
    def _add_all_child_nodes(self, node, used_features):
        node.add_child_nodes(self._all_node_names - used_features)   