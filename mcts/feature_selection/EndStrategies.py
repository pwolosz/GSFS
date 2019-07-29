class EndStrategies:
    """Class for strategies of ending search."""
    
    def __init__(self, name, features_count):
        """
        Parameters
        ----------
        name: str
            Name of the end strategy
        """
        
        self._name = name
        self._features_count = features_count
    
    def are_calculations_over(self, node):
        """
        Method for getting information whether the current iteration of search is over. 
        If the name set in the init is not correct then the exception is thrown.
        Parameters
        ----------
        node: Node
            Current node
        params: dict
            Dictionary containing parameters of MCTS algorithm
        """
        
        if(self._name == 'default'):
            return self._first_new_strategy(node)
        else:
            raise Exception("Error getting end strategy, end strategy \'" + self._name + "\'")
        
    def _first_new_strategy(self, node):
        if node.T > 0:
            return self._features_count == len(node._features)
        return True
