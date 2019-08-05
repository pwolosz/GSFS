import math

class ScoringFunctions():
    """Class used for getting scoring functions that are used to evaluate the node of the MCTS tree."""
    
    def __init__(self, scoring_name, params):
        """
        Parameters
        ----------
        scoring_name: str
            Name of the scoring function that will be used
        params: dict
            Dictionary containing parameters of the scoring functions, please see documentation for more info
        """
        self._params = params
        self._scoring_name = scoring_name

    def set_scoring_function(self, scoring_name, params = None):
        """
        Method for setting scoring function that will be returned from class instance.
        
        Parameters
        ----------
        scoring_name: str
            Name of the scoring function that will be used
        params: dict (default: None)
            Dictionary containing parameters of the scoring functions, please see documentation for more info
            
            """

        self._scoring_name = scoring_name
        if(params is not None):
            self._params = params 

    def get_score(self, parent_node, node, global_scores):
        """
        Method for getting score for selected node. If the name set with init or set_scoring_function is not supported, 
        the exception will be thrown.
        Paramteres
        ----------
        node: Node
            Node which the score will be calculated for
        global_scores: GlobalScores
            Object containing global scores and methods necessary to calculate the score of the node
        """
        
        if(self._scoring_name == 'UCB1'):
            return self._default_scoring(node)
        elif(self._scoring_name == 'UCB1_with_variance'):
            return self._var_scoring(node)
        elif(self._scoring_name == 'UCB1_rave'):
            return self._rave_scoring(parent_node, node, global_scores)
        else:
            raise Exception('Error initializing ScoringFunctions object, \"' + self._name + '\" is not supported.')
     
    def get_new_node_score(self, feature_name, node, global_scores):
        
        if global_scores.get_n(feature_name) == 0:
            return float('Inf')
        
        g_rave = global_scores.get_g_rave_score(feature_name)
        tmp_features = node._features.copy()
        tmp_features.add(feature_name)
        l_rave = global_scores.get_l_rave_score(tmp_features)
        c = self._params['c']
        c_l = self._params['c_l']
        alpha = c/(c + node.T)
        beta = c_l/(c_l + global_scores.get_t_l(tmp_features))
        return (1 - beta) * l_rave + beta * g_rave
    
    def _default_scoring(self, node):
        if(node._parent_node == None or node.T == 0):
            return float("Inf")
        else:
            return node.get_score() + math.sqrt(self._params['c_e'] * math.log(node._parent_node.T)/node.T)
        
    def _var_scoring(self, node):
        if(node._parent_node == None or node.T == 0):
            return float("Inf")
        else:
            return node.get_score() + math.sqrt(self._params['c_e'] * math.log(node._parent_node.T)/node.T)     
        
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
                math.sqrt(c_e * math.log(parent_node.T)/node.T) * 
                min(0.25, node.get_variance() + node.get_variance() + math.sqrt(2 * math.log(parent_node.T)/node.T)))
        