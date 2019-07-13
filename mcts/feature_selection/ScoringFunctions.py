import math

class ScoringFunctions():
    """Class used for getting scoring functions that are used to evaluate the node of the MCTS tree."""
    
    def __init__(self, name, params):
        """
        Parameters
        ----------
        name: str
            Name of the scoring function that will be used
        params: dict
            Dictionary containing parameters of the scoring functions, please see documentation for more info
        """
        self._params = params
        self._name = name

    def set_scoring_function(self, name, params = None):
        """
        Method for setting scoring function that will be returned from class instance.
        
        Parameters
        ----------
        name: str
            Name of the scoring function that will be used
        params: dict (default: None)
            Dictionary containing parameters of the scoring functions, please see documentation for more info
            
            """

        self._name = name
        if(params is not None):
            self._params = params 

    def get_score(self, node, global_scores):
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
        
        if(self._name == 'default'):
            return self._default_scoring(node)
        elif(self._name == 'with_variance'):
            return self._var_scoring(node)
        elif(self._name == 'g_rave'):
            return self._g_rave_scoring(node, global_scores)
        else:
            raise Exception('Error initializing MCTS object, \"' + self._name + '\" is not supported.')
     
    def get_feature_score(self, feature_name, global_scores):
        """
        Methods for gettings score for selected feature.
        Parameters
        ----------
        feature_name: str
            Name of the feature
        global_scores: GlobalScores
            Object containing global scores and methods necessary to calculate the score of the node
        """
        
        return global_scores.get_g_rave_score(feature_name)
    
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
        
    def _g_rave_scoring(self, node, global_scores):
        g_scores = global_scores.scores['g_rave']
        
        if(node._parent_node == None or node.T == 0):
            if(node.feature_name not in g_scores):
                return float("Inf")
            else:
                return g_scores[node.feature_name]['score']
        else:
            c = self._params['c']
            c_l = self._params['c_l']
            alpha = c/(c + node.T)
            beta = c_l/(c_l + node.T)

            return ((1 - alpha) * node.get_score() + 
                alpha * g_scores[node.feature_name]['score'] + 
                math.sqrt(self._params['c_e'] * math.log(node._parent_node.T)/node.T) *
                min(0.25, node.get_variance() + math.sqrt(2 * math.log(node._parent_node.T)/node.T)))      