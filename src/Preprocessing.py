class Preprocessing:
    """Class with preprocessing methods"""
    
    @staticmethod
    def relabel_data(labels, pos_label = None):
        """
        Method for getting transformed labels, which mean changing all positive class labels (pos_label value)
        are changed to 1, other classes are changed to 0.
        
        Parameters
        ----------
        labels: pandas.Series
            List of labels which will be transformed
        pos_label: str|numeric (default: None)
            Value which is positive class indicator, if None then it will be assumed that 1 is positive
        
        Return: list containing values 0 and 1 with same length as labels 
        """

        if pos_label is None:
            pos_label = max(labels)
        with pd.option_context('mode.chained_assignment', None):
            labels.loc[labels != pos_label] = 0
            labels.loc[labels == pos_label] = 1

        return labels