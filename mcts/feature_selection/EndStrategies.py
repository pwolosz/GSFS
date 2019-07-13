class EndStrategies:
    """Class for strategies of ending search."""
    
    def __init__(self, name):
        """
        Parameters
        ----------
        name: str
            Name of the end strategy
        """
        
        self._name = name
    
    def are_calculations_over(self, node, params):
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
            return self._first_new_strategy(node, params)
        else:
            raise Exception("Error getting end strategy, end strategy \'" + self._name + "\'")
        
    def _first_new_strategy(self, node, params):
        print("_first_new_strategy:")
        if(node.T > 0 and not node._is_subtree_full):
            print("first if")
            return False
        else:
            if(node._parent_node == None):
                print("second if, no parent node")
                return False
            else:
                print("else")
                return True
