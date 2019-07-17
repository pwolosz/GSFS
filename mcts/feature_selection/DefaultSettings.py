class DefaultSettings:
    """Class containing default settings for MCTS."""
    
    @staticmethod
    def get_default_params():
        """Method for getting default settings."""
        
        return {
            "c_e": 2,
            "c": 1,
            "c_l": 1,
            "cv": 4,
            "b_T": 0.5,
            "not_visited_score_inf": True
        }
    
    @staticmethod
    def merge_params(params):
        """
        Method for merging user's params with the default ones
        Parameters
        ----------
        params: dict
            Dictioary with mcts parameters
        """
        
        merged_params = DefaultSettings.get_default_params()
        for key, value in params.items():
            merged_params[key] = value
            
        return merged_params