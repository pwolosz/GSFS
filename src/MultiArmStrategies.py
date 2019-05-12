class MultiArmStrategies:
    """Class for representing strategies used for selecting next node to visit in MCTS. 
    For more info please see documentation"""
    
    def __init__(self, name, all_node_names):
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
        
    def multiarm_strategy(self, node, used_features, scoring_function, other_scores):
        """
        Method for getting next node in current search.
        Parameters
        ----------
        node: Node
            Current node
        used_features: list
            List of feature names that are already used in search path
        scoring_function: ScoringFunctions
            Scoring function that is used in current search instance
        other_scores: dict
            Dictionary containing scores used in strategy
        """
        
        if(self._name == 'default'):
            return self._default_strategy(node, used_features, scoring_function, other_scores)
        else:
            raise Exception("Error getting multiarm strategy, strategy \'" + self._name + "\' is not supported.")   
    
    def _default_strategy(self, node, used_features, scoring_function, other_scores):
        #print("default strategy: " + node.feature_name)
        if(len(node.child_nodes) == 0):
            #print("first if")
            self._add_child_nodes(node, used_features)
            return node.child_nodes[0]
        else:
            #print("else")
            best_score = 0
            best_node = None
            tmp_score = 0
            
            for child_node in node.child_nodes:
                if(not child_node._is_subtree_full):
                    score = scoring_function(child_node, other_scores)
                    if(score > best_score):
                        best_score = score
                        best_node = child_node
            return best_node
  
    def _add_child_nodes(self, node, used_features):
        #print('adding nodes to ' + node.feature_name + ' :' + ' '.join(self._all_node_names - used_features))
        node.add_child_nodes(self._all_node_names - used_features)   