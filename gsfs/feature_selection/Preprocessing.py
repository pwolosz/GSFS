import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Preprocessing:
    """Class with preprocessing methods"""
    
    @staticmethod
    def relabel_data(labels, pos_label = None):
        """
        Method for getting transformed labels, which means changing all positive class labels (pos_label value)
        are changed to 1, other classes are changed to 0.
        
        Parameters
        ----------
        labels: pandas.Series
            List of labels which will be transformed
        pos_label: str|numeric (default: None)
            Value which is positive class indicator, if None then it will be assumed that 1 is positive
        
        Return: list containing values 0 and 1 with same length as labels.
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
        Method for performing one-hot encoding. Using LabelEncoder and OneHotEncoder from sklearn.preprocessing.
        All columns containing string values will be encoded, if the column has n unique values then n new columns will
        be added, and the end original column is deleted.

        Parameters
        ----------
        data: pandas.DafaFrame
        Dataframe that will be transformed

        Return: pandas.DataFrame with encoded columns.
        """

        for col in data:
            if(isinstance(data.loc[0,col], str)):
                dummies = pd.get_dummies(data.loc[:,col])
                dummies.columns = col + '_' + dummies.columns
                data = data.join(dummies)
                data = data.drop(columns = col)

        return data