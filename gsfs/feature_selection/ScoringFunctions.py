import math

class ScoringFunctions():
    """
    Class containing scoring functions that are used during graph search. 
    Available scoring functions are "UCB1", "UCB1_with_variance" and "UCB1_rave".
    """
    
    def __init__(self, scoring_name, params):
        """
        Parameters
        ----------
        scoring_name: str
            Name of the scoring function that will be used,
        params: dict
            Parameters of scoring functions.
        """
        self._params = params
        self._scoring_name = scoring_name

    def set_scoring_function(self, scoring_name, params = None):
        """
        Method setting scoring function.
        
        Parameters
        ----------
        scoring_name: str
            Name of the scoring function that will be used,
        params: dict (default: None)
            Parameters of scoring functions, if None then the one specified during initialization will be used.
            
        """

        self._scoring_name = scoring_name
        if params is not None :
            self._params = params 

    def get_score(self, parent_node, node, global_scores):
        """
        Method for getting score of the node.

        Paramteres
        ----------
        node: gsfs.feature_selection.Node
            Current node in search iteration,
        global_scores: gsfs.feature_selection.GlobalScores
            Object containing values that are needed in calculating score of the node, e.g RAVE.

        Returns: float
            Score of the node.
        """
        
        if self._scoring_name == 'UCB1':
            return self._ucb_scoring(parent_node, node)
        elif self._scoring_name == 'UCB1_with_variance':
            return self._ucb_var_scoring(parent_node, node)
        elif self._scoring_name == 'UCB1_rave':
            return self._rave_scoring(parent_node, node, global_scores)
        else:
            raise Exception('Error initializing ScoringFunctions object, \"' + self._scoring_name + '\" is not supported.')
     
    def get_new_node_score(self, feature_name, node, global_scores): 
        """
        Method for getting score for new node (node that hasn’t been added to the graph). New node would have set of features Node.F+{feature_name}.

        Parameters
        ----------
        feature_name: str
            Name of the feature that is added to current node’s set of features,
        node: gsfs.feature_selection.Node
            Current node in search for which adding the new node is considered,
        global_scores: gsfs.feature_selection.GlobalScores
            Object containingvalues that are needed in calculating score of the node, e.g RAVE.
			
        Returns: float
            Score of the node.
        """
        
        if global_scores.get_n(feature_name) == 0:
            return float('Inf')
        
        g_rave = global_scores.get_g_rave_score(feature_name)
        tmp_features = node._features.copy()
        tmp_features.add(feature_name)
        l_rave = global_scores.get_l_rave_score(tmp_features)
        c = self._params['c']
        c_l = self._params['c_l']
        beta = c_l/(c_l + global_scores.get_t_l(tmp_features))
        return (1 - beta) * l_rave + beta * g_rave
    
    def _ucb_scoring(self, parent_node, node):
        if parent_node == None or node.T == 0:
            return float("Inf")
        else:
            return node.get_score() + math.sqrt(self._params['c_e'] * math.log(parent_node.T)/node.T)
        
    def _ucb_var_scoring(self, parent_node, node):
        if parent_node == None or node.T == 0:
            return float("Inf")
        else:
            return node.get_score() + math.sqrt((self._params['c_e'] * math.log(parent_node.T)/node.T) * 
            min(0.25, node.get_variance() + math.sqrt(2 * math.log(parent_node.T)/node.T)))     
        
    def _rave_scoring(self, parent_node, node, global_scores):
        t_l = global_scores.get_t_l(node._features)
        c = self._params['c']
        c_l = self._params['c_l']
        c_e = self._params['c_e']
        alpha = c/(c + node.T)
        beta = c_l/(c_l + t_l)
        new_feature = node._features.difference(parent_node._features).pop()
        
        return ((1 - alpha) * node.get_score() + 
                alpha * ((1 - beta) * global_scores.get_l_rave_score(node._features) + beta * global_scores.get_g_rave_score(new_feature)) +
                math.sqrt((c_e * math.log(parent_node.T)/node.T) * 
                min(0.25, node.get_variance() + math.sqrt(2 * math.log(parent_node.T)/node.T))))
        