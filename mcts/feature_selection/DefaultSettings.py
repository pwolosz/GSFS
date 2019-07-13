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
            "b_T": 0.5
        }
    