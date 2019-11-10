class DefaultSettings:
    """Class containing static methods for getting algorithm parameters"""
    
    @staticmethod
    def get_default_params():
        """
        Method for getting default parameters.

        Returns: dict
        Dictionary containing default parameters
        """
        
        return {
            "c_e": 2,
            "c": 1,
            "c_l": 1,
            "cv": 4,
            "b_T": 0.5,
            "test_size": 0.25,
            "new_node_preference": 1
        }
    
    @staticmethod
    def merge_params(params):
        """
        Method for merging user’s parameters with the default ones,
        the input from user will override the default ones.

        Parameters
        ----------
        params: dict
            Dictionary containing parameters defined by the user.

        Returns: dict
            Dictionary containing merged parameters.
        """
        
        merged_params = DefaultSettings.get_default_params()
        for key, value in params.items():
            merged_params[key] = value
            
        print('Parameters:')
        print(merged_params)

        if merged_params['c_e'] < 0:
            raise Exception('c_e must be >= 0')
        if merged_params['c'] < 0:
            raise Exception('c must be >= 0')
        if merged_params['c_l'] <= 0:
            raise Exception('c_l must be > 0')
        if merged_params['cv'] <= 0:
            raise Exception('cv must be > 0')
        if merged_params['b_T'] <= 0:
            raise Exception('b_T must be > 0')
        if merged_params['test_size'] <= 0:
            raise Exception('test_size must be > 0')
        if merged_params['new_node_preference'] <= 0:
            raise Exception('new_node_preference must be > 0')
            
        return merged_params