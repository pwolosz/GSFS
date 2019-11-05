from gsfs.feature_selection.Node import *
from gsfs.feature_selection.DefaultSettings import *
from gsfs.feature_selection.CV import *
from gsfs.feature_selection.Preprocessing import *
from gsfs.feature_selection.MultiArmStrategies import *
from gsfs.feature_selection.EndStrategies import *
from gsfs.feature_selection.ScoringFunctions import *
from gsfs.feature_selection.BuildInMetrics import *
from gsfs.feature_selection.GlobalScores import *
from gsfs.feature_selection.NodeAdder import *
from gsfs.feature_selection.DrawTree import draw_tree
from gsfs.feature_selection.TrainTestScore import *

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import pandas as pd

class GSFS:
    """Class representing object used for Graph Based Feature Selection."""
    def __init__(self, 
                 model,
                 calculations_budget,
                 calculations_done_condition = 'iterations',
                 params = None,
                 metric = 'roc_auc', 
                 scoring_function = 'UCB1_rave', 
                 multiarm_strategy = 'discrete', 
                 end_strategy = 'default',
                 with_cv = False,
                 preprocess = True):
        """
        Parameters
        ----------
        model: sklearn model
            Model that implements fit, predict, predict_proba methods, used during feature selection - GSFS is a wrapper around that model,
        calculations_budget: int
            Budget for calculations, it can be either time in seconds or number of iterations,
        calculations_done_condition: str (default: iterations)
            Information of what type of budget the algorithm considers (available values are "iterations" and "time"), 
        params: dict (default: None)
            Dictionary containing possible parameters of algorithm, the values from this dictionary will be taken
            as overrides of default values of the parameters (DefaultSettings.get_default_params()), if nothing is provided
            then all parameters will have default values,
        metric: str (default: "roc_auc")
            Name of the metric that will be used for scoring the model's predictions, it has to be one of the metrics 
            from gsfs.feature_selection.BuildInMetrics ("roc_auc", "acc", "f1"),
        scoring_function: str (default: "UCB1_rave")
            Name of the scoring function used during search of the graph, possible values are "UCB1_rave", "UCB1_with_variance", "UCB1",
        multiarm_strategy: str (default: "discrete")
            Name of the strategy that will be used during adding new node to the graph, possible values are "discrete" and "continuous", 
        end_strategy: str (default: "default")
            Name of the end strategy, default one is stop on first new node,
        with_cv: boolean (default: False)
            Information whether use cross-validation during calculating model's score, if not then train-test score will be used,
        preprocess: boolean (default: True)
            Information whether use the preprocessing of input data, meaning resetting index of data and labels
            relabeling the labels to 0 and 1.
        """
        
        
        self._metric_name = metric
        self._scoring_function_name = scoring_function
        self._multiarm_strategy_name = multiarm_strategy
        self._end_strategy_name = end_strategy
        self._best_features = None
        self._best_score = 0
        self._feature_names = None
        self._calculations_done_condition = calculations_done_condition
        self._calculations_budget = calculations_budget
        self._model = clone(model)
        self._time = 0
        self._iterations = 0
        self._preprocess = preprocess
        self._with_cv = with_cv
        
        print('Using cross-validation: ' + str(with_cv))
        
        if params is None:
            print('No param overrides provided, using default ones')
            self._params = DefaultSettings.get_default_params()
        else:
            self._params = DefaultSettings.merge_params(params)
        
    @staticmethod 
    def print_info():
        print('Graph Search Feature Selection')
        print('Version: 1.0.0')
    
    def fit(self, data, out_variable, pos_class = 'numeric', warm_start = False, 
                 calculations_done_conditions = None,
                 calculations_budget = None):
        """
        Method for perfoming the fitting of the feature selection algorithm.
        
        Parameters
        ----------
        data: pandas.DataFrame
            Dataset that will be used for fitting, containing all features except the output variable,
        out_variable: pandas.Series
            Series containing output variable of the dataset,
        pos_class: str (default: 'numeric')
            Value indicating the positive class, which will be transformed to 1 and all other values from output variable
            will be 0, by default the highest value of the output variable will be taken as positive class,
        warm_start: boolean (default: False)
            Information whether before the actual graph based feature selection the g-RAVE scores will be initialised from
            feature_importance of RandomForestClassifier,
        calculations_done_condition: str (default: None)
            Information of what type of budget the algorithm considers (available values are 'iterations' and 'time'), 
            default value is taken from constructor,
        calculations_budget: int (default: None)
            Budget for calculations, it can be either time in seconds or number of iterations, default value is taken
            from constructor.

        Returns: None.
        """
        
        self._pos_class = pos_class
        if calculations_done_conditions is not None:
            self._calculations_done_condition = calculations_done_conditions
        
        if calculations_budget is not None:
            self._calculations_budget = calculations_budget
            
        data, out_variable = self._preprocess_input(data, out_variable)
        
        self._classification_fit_start(data, out_variable, warm_start)
    
    def refit(self, data, out_variable, calculations_budget):
        """Not fully supported method, only for experimenting purposes."""
        
        data, out_variable = self._preprocess_input(data, out_variable)
        
        if self._calculations_done_condition == 'iterations':
            self._calculations_budget += calculations_budget + 1
        else:
            self._calculations_budget = calculations_budget
            
        self._classification_fit(data, out_variable)
    
    def _preprocess_input(self, data, out_variable):
        data = data.reset_index(drop=True)
        data.columns = [str(col) for col in data.columns]
        
        out_variable = out_variable.reset_index(drop=True)
        
        if self._preprocess:
            out_variable = self._preprocess_labels(out_variable, self._pos_class)
            
        return data, out_variable
    
    def _classification_fit_start(self, data, out_variable, warm_start):
        self._init_fitting_values(data)
        
        if warm_start:
            rf = RandomForestClassifier()
            rf.fit(data, out_variable)
            for i in range(len(data.columns)):
                self._global_scores._update_g_rave_score([data.columns[i]], rf.feature_importances_[i])

        self._classification_fit(data, out_variable)
    
    def _classification_fit(self, data, out_variable):
        self._time = time.time()
        
        while not self._is_fitting_over():
            self._single_classification_iteration(data, out_variable)
        
        self._model.fit(data.loc[:, self._best_features], out_variable)
    
    def _single_classification_iteration(self, data, out_variable):
        used_nodes = [None]*(self._longest_graph_branch+1)
        node = self._root
        used_nodes[0] = node
        used_nodes_index = 1
        is_iteration_over = False
        while not is_iteration_over:
            node = self._multiarm_strategy.multiarm_strategy(node, self._scoring_functions, 
                                                             self._global_scores, self._node_adder)
            is_iteration_over = self._end_strategy.are_calculations_over(node)
            used_nodes[used_nodes_index] = node
            used_nodes_index += 1  

        score = self._get_score_for_features(data[list(node._features)], out_variable)
        self._update_nodes(used_nodes, score)
        self._global_scores.update_score(node._features, score)
        
        
        if score > self._best_score:
            self._best_score = score
            self._best_features = list(node._features)
            self._scores_history = self._scores_history.append({
                'score': score, 
                'features': self._best_features,
                'time': time.time() - self._time,
                'iteration': self._iterations
            },ignore_index=True)
        
        if self._longest_graph_branch < used_nodes_index:
            self._longest_graph_branch = used_nodes_index

    
    def _update_nodes(self, used_nodes, score):
        for i in range(len(used_nodes)):
            if used_nodes[i] is None:
                return
            used_nodes[i].add_score(score)
    
    def _get_score_for_features(self, data, out_variable, with_cv = None, model = None):
        if with_cv is None:
            with_cv = self._with_cv
        
        if model is None:
            model = clone(self._model)
        
        if with_cv:
            return CV.cv(self._metric, self._metric_name, model, data, out_variable, self._params['cv'])
            
        return TrainTestScore.train_test_score(self._metric, self._metric_name, 
                                                   model, data, out_variable, self._params['test_size'])
            
    
    def _init_fitting_values(self, data):
        self._root = Node(set(), None)
        self._feature_names = set(data.columns)
        self._multiarm_strategy = MultiArmStrategies(self._multiarm_strategy_name, self._feature_names, self._params)
        self._end_strategy = EndStrategies(self._end_strategy_name, len(self._feature_names))
        self._scoring_functions = ScoringFunctions(self._scoring_function_name, self._params)
        self._metric = BuildInMetrics().get_metric(self._metric_name)
        self._best_features = None
        self._best_score = 0
        self._longest_graph_branch = 1
        self._global_scores = GlobalScores()
        self._scores_history = pd.DataFrame(columns=['score','features','time','iteration'])
        self._node_adder = NodeAdder(self._root)
        self._time = time.time()
        self._iterations = 0
    
    def _is_fitting_over(self):
        self._iterations += 1
        self._print_calculations_info_if_needed()
        
        if self._calculations_done_condition == 'iterations':
            return self._iterations > self._calculations_budget 
        else:
            return (time.time() - self._time) > self._calculations_budget 
    
    def _print_calculations_info_if_needed(self):
        if self._calculations_done_condition != 'iterations':
            return
        
        calc_interval = self._calculations_budget/100
        
        if (int((self._iterations/calc_interval))-int(((self._iterations-1)/calc_interval))) > 0:
            print(str(self._iterations - 1) + '/' + str(self._calculations_budget))
    
    def get_best_features(self):
        """
        Method for getting best features from search (features from best node in the graph).

        Returns: list
            List with best features.
        """
        
        return self._best_features
    
    def get_best_score(self):
        """
        Method for getting score for best node (best node in graph).

        Returns: float
            Score of the best node.
        """
        
        return self._best_score
    
    def get_search_history(self):
        """
        Method for getting search history.

        Returns: pandas.DataFrame
            Data frame containing best found nodes since the beginning of the search. 
        """
        
        return self._scores_history
    
    def _preprocess_labels(self, labels, pos_class):
        if pos_class == 'numeric':
            pos_class = 1
            
        return Preprocessing.relabel_data(labels, pos_class)
    
    def predict(self, data):
        """
        Method for predicting classes for input data.

        Parameters
        ----------
        data: pandas.DataFrame
            Data for which the classes will be returned.

        Returns: list
            List with predictions, every value is either 1 (positive) or 0 (negative) class, i-th row is a class for i-th input observation.
        """

        data.columns = [str(col) for col in data.columns]
        return self._model.predict(data.loc[:, self._best_features])
    
    def predict_proba(self, data):
        """
        Method for getting probabilities of classes for input data.

        Parameters
        ----------
        data: pandas.DataFrame
            Data for which probabilites will be returned.

        Returns: 2-dimensional list
            List, where first column represents probabilities of 0 (negative) class and second probabilities of 1 (positive) class.
        """

        data.columns = [str(col) for col in data.columns]
        return self._model.predict_proba(data.loc[:, self._best_features])
    
    def get_features_importances(self):
        """
        Method for getting importances (g-RAVE) of all features used in algorithm.
        
        Returns: dict
            Dictionary containing pairs of features and importances.
        """
        importances = dict([k, self._global_scores.get_g_rave_score(k)] for k,v in self._global_scores.scores['g_rave'].items())
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse = True))
    
    def one_hot_encode(self, data):
        """
        Method for one-hot encoding the data.
        
        Parameters
        ----------
        data: pandas.DataFrame
            Input dataset that will be one-hot encoded.
            
        Returns: pandas.DataFrame
            One-hot encoded dataset.
        """
        
        data.columns = [str(col) for col in data.columns]
        return Preprocessing.one_hot_encode(data)
    
    def draw_graph(self, file_name = None, view = True, view_nodes_info = False):
        """
        Method for drawing the graph of the search algorithm.
        
        Parameters
        ----------
        file_name: str (default: None)
            If other than None then search graph will be saved with file_name (can contain whole path),
        view: boolean (default: True)
            If True then graph will be displayed in default browser,
        view_nodes_info: boolean (default: False)
            If True then average score, number of visits and variance of scores is displayed on every node.
            
        Returns: None
        """
        
        draw_tree(self._node_adder, file_name, view, view_nodes_info)
    
    def save_stats_to_files(self, path):
        """
        Method for saving the stats of current run of feature selection task to file.
        
        Parameters
        ----------
        path: str
            Path of the file to which the stats will be saved.
        
        Returns: None
        """
        if self._best_features is None:
            raise('Model not trained, please fit the model first')
           
        with open((path + '.txt'), 'w') as f:
            f.write('Best score: ' + str(self._best_score))
            f.write('\nBest features: ' + ', '.join(self._best_features))
            f.write('\nLongest branch: ' + str(self._longest_graph_branch))
            f.write('\nMetric name: ' + str(self._metric_name))
            f.write('\nScoring function: ' + str(self._scoring_function_name))
            f.write('\nMultiarm strategy: ' + str(self._multiarm_strategy_name))
            f.write('\nEnd strategy: ' + str(self._end_strategy_name))
            f.write('\nCalculations done condition: ' + str(self._calculations_done_condition))
            f.write('\nCalculations budget: ' + str(self._calculations_budget))
            f.write('\nWith cv: ' + str(self._with_cv))
            
            f.write('\nScores history: ')
            
            f.write('\nParameters: ')
            for key, value in self._params.items():
                f.write('\n' + key + ': ' + str(value))
                
        self.get_search_history().to_csv(path + '.csv')
        self._global_scores.get_g_rave_dataframe().to_csv(path + '_g_rave.csv')
        self._global_scores.get_l_rave_dataframe().to_csv(path + '_l_rave.csv')
        
    def get_best_model(self, data, labels, with_cv = None, model = None, preprocess = True):
        """
        Method for getting best model for selected dataset. Features are selected using greedy approach
        based on g-RAVE, model is trained on best feature, then on two best features, until all features
        are used.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataset for which the model will be returned,
        labels: pandas.Series
            Labels of the dataset,
        with_cv: boolean (default: None)
            Setting that if True, then cross-validation will be used when selecting the best features for the model, 
            otherwise train-test split will be used, if None then setting from input parameters of GSFS object is taken,
        model: sklearn model (default: None)
            Model for which best features will be selected, if None then copy of the model from GSFS object is taken,
        preprocess: boolean (default: True)
            Information whether use the preprocessing of input data, meaning resetting index of data and labels
            relabeling the labels to 0 and 1.
            
        Returns: dict
            Dictionary containing entries: "model" (model trained on best features), "scores" (data frame containing 
            scores of all sets of features used in greedy search) and "best_features" (best features on which final model
            was trained).
        """

        used_features = []
        scores = []
        features_str = []
        best_score = 0
        best_features = []

        if model is None:
            model = clone(self._model)
        
        if preprocess:
            data, labels = self._preprocess_input(data, labels)
        
        for key in self.get_features_importances().keys():
            used_features.append(key)
            features_str.append(','.join(used_features))
            score = self._get_score_for_features(data.loc[:,used_features],labels,with_cv,model)
            
            if best_score < score:
                best_score = score
                best_features = used_features.copy()
                
            scores.append(score)
            
        print('Found best model with score ' + str(best_score) + ', refitting')
        new_model = clone(model)
        new_model.fit(data.loc[:,best_features],labels)
        
        return {'model': new_model,
               'scores': pd.DataFrame({'features': features_str, 'score': scores}),
               'best_features': best_features}