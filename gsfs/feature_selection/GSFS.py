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

class GSFS:
    """Class for perfoming Graph Search Feature Selection"""
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
            Model that implements fit, predict, predict_proba methods, used during feature selection - GSFS is a wrapper around that model
        calculations_budget: numeric
            Budget for calculations, it can be either time in seconds or number of iterations
        calculations_done_condition: str
            Information of what type of budget the algorithm consideres (available values are 'iterations' and 'time'), 
            default value is 'iterations'
        params: dict
            Dictionary containing possible parameters of algorithm, the values from this dictionary will be taken
            as overrides of default values of the parameters (DefaultSettings.get_default_params()), if nothing is provided
            then all parameters will have default values
        metric: str
            Name of the metric that will be used for scoring the model's predictions, it has to be one of the metrics 
            from BuildInMetrics ('roc_auc', 'acc,', 'f1'), default value is 'roc_auc'
        scoring_function: str
            Name of the scoring function used during search of the graph, possible values are 'UCB1_rave', 'UCB1_with_variance',
            'UCB1', default value is 'UCB1_rave'
        multiarm_strategy: str
            Name of the strategy that will be used during adding new node to the graph, possible values are 'discrete' and
            'continuous', default value is 'discrete'
        end_strategy: str
            Name of the end strategy, default one is stop on first new node
        with_cv: boolean
            Information whether use cross-validation during calculating model's score, if not then train-test score will be used,
            default value is False
        preprocess: boolean
            Information whether use the preprocessing of input data, meaning applying one-hot encoding to the features and
            relabeling the labels to 0 and 1
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
        self._model = model
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
        
    def fit(self, data, out_variable, pos_class = 'numeric', warm_start = False, 
                 calculations_done_conditions = None,
                 calculations_budget = None):
        
        """
        Method for perfoming the fitting of the feature selection algorithm.
        
        Parameters
        ----------
        data: pd.DataFrame
            Dataset that will be used for fitting contiaing all features except the output variable
        out_variable: pd.Series
            Series containing output variable of the dataset
        pos_class: str
            Value indicating the positive class, which will be transformed to 1 and all other values from output variable
            will be 0, by default the highest value of the output variable will be taken as positive class
        warm_start: boolean
            Information whether before the actual graph based feature selection the g-RAVE scores will be initialised from
            feature_importance of RandomForestClassifier
        calculations_done_condition: str
            Information of what type of budget the algorithm consideres (available values are 'iterations' and 'time'), 
            default value is taken from constructor
        calculations_budget: numeric
            Budget for calculations, it can be either time in seconds or number of iterations, default value is taken
            from constructor
        """
        
        self._pos_class = pos_class
        if calculations_done_conditions is not None:
            self._calculations_done_condition = calculations_done_conditions
        
        if calculations_budget is not None:
            self._calculations_budget = calculations_budget
            
        data, out_variable = self._preprocess_input(data, out_variable)
        
        self._classification_fit_start(data, out_variable, warm_start)
    
    def refit(self, data, out_variable, calculations_budget):
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
        used_nodes = [None]*(self._longest_tree_branch+1)
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
        
        if self._longest_tree_branch < used_nodes_index:
            self._longest_tree_branch = used_nodes_index

    
    def _update_nodes(self, used_nodes, score):
        for i in range(len(used_nodes)):
            if used_nodes[i] is None:
                return
            used_nodes[i].add_score(score)
    
    def _get_score_for_features(self, data, out_variable):
        if self._with_cv:
            return CV.cv(self._metric, self._metric_name, self._model, data, out_variable, self._params['cv'])
            
        return TrainTestScore.train_test_score(self._metric, self._metric_name, 
                                                   self._model, data, out_variable, self._params['test_size'])
            
    
    def _init_fitting_values(self, data):
        self._root = Node(set(), None)
        self._feature_names = set(data.columns)
        self._multiarm_strategy = MultiArmStrategies(self._multiarm_strategy_name, self._feature_names, self._params)
        self._end_strategy = EndStrategies(self._end_strategy_name, len(self._feature_names))
        self._scoring_functions = ScoringFunctions(self._scoring_function_name, self._params)
        self._metric = BuildInMetrics().get_metric(self._metric_name)
        self._best_features = None
        self._best_score = 0
        self._longest_tree_branch = 1
        self._global_scores = GlobalScores()
        self._scores_history = pd.DataFrame(columns=['score','features','time','iteration'])
        self._node_adder = NodeAdder(self._root)
        self._time = time.time()
    
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
            print(str(self._iterations) + '/' + str(self._calculations_budget))
    
    def get_best_features(self):
        return self._best_features
    
    def get_best_score(self):
        return self._best_score
    
    def get_search_history(self):
        return self._scores_history
    
    def _preprocess_labels(self, labels, pos_class):
        if pos_class == 'numeric':
            pos_class = 1
            
        return Preprocessing.relabel_data(labels, pos_class)
    
    def predict(self, data):
        data.columns = [str(col) for col in data.columns]
        return self._model.predict(data.loc[:, self._best_features])
    
    def predict_proba(self, data):
        data.columns = [str(col) for col in data.columns]
        return self._model.predict_proba(data.loc[:, self._best_features])
    
    def get_features_importances(self):
        return dict([k, self._global_scores.get_g_rave_score(k)] for k,v in self._global_scores.scores['g_rave'].items())
    
    def one_hot_encode(self, data):
        data.columns = [str(col) for col in data.columns]
        return Preprocessing.one_hot_encode(data)
    
    def draw_tree(self, file_name = None, view = True, view_nodes_info = False):
        draw_tree(self._node_adder, file_name, view, view_nodes_info)
    
    def save_stats_to_files(self, path):
        """
        Method for saving the stats of current run of feature selection task to file
        
        Parameters
        ----------
        path: str
            Path of the file to which the stats will be saved
        """
        if self._best_features is None:
            raise('Model not trained, please fit the model first')
           
        with open((path + '.txt'), 'w') as f:
            f.write('Best score: ' + str(self._best_score))
            f.write('\nBest features: ' + ', '.join(self._best_features))
            f.write('\nLongest branch: ' + str(self._longest_tree_branch))
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