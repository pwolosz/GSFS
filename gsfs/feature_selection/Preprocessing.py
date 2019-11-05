import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Preprocessing:
    """Class with preprocessing methods"""
    
    @staticmethod
    def relabel_data(labels, pos_label = None):
        """
        method for relabeling input data, changing value specified as "pos_label" to 1 and all other values to 0.
        
        Parameters
        ----------
        labels: pandas.Series
            Out variable of the dataset, will be relabeled to 1 (positive class) and 0 (the rest of the classes),
        pos_label: str|numeric (default: None)
            Positive class label, if None then maximum value of all values present in "labels" will be taken as positive label.
        
        Returns: pandas.Series
            Relabeled variable.
        """

        if pos_label is None:
            pos_label = max(labels)
        with pd.option_context('mode.chained_assignment', None):
            labels.loc[labels != pos_label] = 0
            labels.loc[labels == pos_label] = 1

        return labels

    @staticmethod
    def one_hot_encode(data):
        """
        Method that can be used to one-hot encode input data. All columns containing string not-numerical will be one-hot encoded.

        Parameters
        ----------
        data: pandas.DafaFrame
            Data frame that will be one-hot encoded.

        Returns: pandas.DataFrame 
            Data frame with one-hot encoding.
        """

        for col in data:
            ind = -1

        for i in data.index.values:
            if not pd.isnull(data.loc[i,col]):
                ind = i

        if ind == -1:
            print('Column "' + col + '" contains only NaN values, removing it')
            data = data.drop(columns=[col])
        else:   
            if isinstance(data.loc[ind,col], str):
                dummies = pd.get_dummies(data.loc[:,col])
                dummies.columns = col + '_' + dummies.columns
                data = data.join(dummies)
                data = data.drop(columns = col)
                
        return data