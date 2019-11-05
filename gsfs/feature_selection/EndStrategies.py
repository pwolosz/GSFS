class EndStrategies:
    """
    Class containing strategies for ending an iteration of graph search. 
    Only one end strategy issupported - iteration is stopped 
    if the current node is a new node (T=0) or the node containsall possible features.
    """
    
    def __init__(self, name, features_count):
        """
        Parameters
        ----------
        name: str
            Name of the end strategy that will be used, currently only default is supported.
        features_count: int
            Number of all features in the dataset.
        """
        
        self._name = name
        self._features_count = features_count
    
    def are_calculations_over(self, node):
        """
        Method for getting information whether the current search iterationmust be finished.

        Parameters
        ----------
        node: gsfs.feature_selection.Node
            Current node in search iteration.
        
        Returns: boolean
            Value indicating whether the iteration should end or not.
        """
        
        if(self._name == 'default'):
            return self._first_new_strategy(node)
        else:
            raise Exception("Error getting end strategy, end strategy \'" + self._name + "\'")
        
    def _first_new_strategy(self, node):
        if node.T > 0:
            return self._features_count == len(node._features)
        return True
