from mcts.feature_selection.Node import *
from mcts.feature_selection.DefaultSettings import *
from mcts.feature_selection.CV import *
from mcts.feature_selection.Preprocessing import *
from mcts.feature_selection.MultiArmStrategies import *
from mcts.feature_selection.EndStrategies import *
from mcts.feature_selection.ScoringFunctions import *
from mcts.feature_selection.BuildInMetrics import *
from mcts.feature_selection.GlobalScores import *
import time
from sklearn.ensemble import RandomForestClassifier

class MCTS:
    """Class for MCTS"""
    def __init__(self, 
                 model,
                 task = 'classification',
                 calculations_done_condition = 'iterations',
                 calculations_budget = 10,
                 params = None,
                 metric = 'acc', 
                 scoring_function = 'g_rave', 
                 multiarm_strategy = 'discrete', 
                 end_strategy = 'default'):
        
        self._metric_name = metric
        self._scoring_function_name = scoring_function
        self._multiarm_strategy_name = multiarm_strategy
        self._end_strategy_name = end_strategy
        self._best_features = None
        self._best_score = 0
        self._task = task
        self._root = Node("")
        self._feature_names = None
        self._calculations_done_condition = calculations_done_condition
        self._calculations_budget = calculations_budget
        self._model = model
        self._time = 0
        self._iterations = 0
        
        if params is None:
            self._params = DefaultSettings.get_default_params()
        else:
            self._params = DefaultSettings.merge_params(params)
        
    def fit(self, data, out_variable, preprocess = True, pos_class = 'numeric', warm_start = True,
                 calculations_done_conditions = None,
                 calculations_budget = None):
        
        if calculations_done_conditions is not None:
            self._calculations_done_condition = calculations_done_conditions
        
        if calculations_budget is not None:
            self._calculations_budget = calculations_budget
            
        data = data.reset_index(drop=True)
        data.columns = [str(col) for col in data.columns]
        out_variable = out_variable.reset_index(drop=True)

        if preprocess:
            out_variable = self._preprocess_labels(out_variable, pos_class)
        
        if self._task == 'classification':
            self._classification_fit(data, out_variable, warm_start)
        else:
            self._regression_fit(data, out_variable)
            
    def _classification_fit(self, data, out_variable, warm_start):
        self._init_fitting_values(data)
        
        if warm_start:
            rf = RandomForestClassifier()
            rf.fit(data, out_variable)
            for i in range(len(data.columns)):
                self._global_scores.update_g_rave_score(data.columns[i], rf.feature_importances_[i])
        
        while not self._is_fitting_over():
            self._single_classification_iteration(data, out_variable)
        
        self._model.fit(data.loc[:, self._best_features], out_variable)
    
    def _regression_fit(self, data, out_variable):
        return None
    
    def _single_classification_iteration(self, data, out_variable):
        used_features = set()
        node = self._root
        is_iteration_over = False
        while not is_iteration_over:
            node = self._multiarm_strategy.multiarm_strategy(node, used_features, self._scoring_functions, self._global_scores,self._params)
            is_iteration_over = self._end_strategy.are_calculations_over(node, self._params)
            used_features.add(node.feature_name)
        
        score = CV.cv(self._metric, self._metric_name, self._model, data, out_variable, self._params['cv'])
        node.update_scores_up(score, self._global_scores)
        if(score > self._best_score):
            self._best_score = score
            self._best_features = used_features
            self._scores_history.append({'score': score, 'features': used_features})
        
        if(self._longest_tree_branch < len(used_features)):
            self._longest_tree_branch = len(used_features)

    
    def _single_regression_iteration(self):
        return None
    
    def _init_fitting_values(self, data):
        self._feature_names = set(data.columns)
        self._multiarm_strategy = MultiArmStrategies(self._multiarm_strategy_name, self._feature_names, self._params)
        self._end_strategy = EndStrategies(self._end_strategy_name)
        self._scoring_functions = ScoringFunctions(self._scoring_function_name, self._params)
        self._metric = BuildInMetrics().get_metric(self._metric_name)
        self._best_features = None
        self._best_score = 0
        self._iterations = 0
        self._longest_tree_branch = 0
        self._time = time.time()
        self._global_scores = GlobalScores()
        self._scores_history = []
    
    def _is_fitting_over(self):
        if(self._root._is_subtree_full):
            print('Whole tree searched, finishing prematurely')
            return True
        
        if(self._calculations_done_condition == 'iterations'):
            self._iterations += 1
            return self._iterations > self._calculations_budget 
        else:
            return (time.time() - self._time) > self._calculations_budget 
    
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
    
    def get_number_of_iterations(self):
        return self._global_scores.scores['g_rave']['']['n']
    
    def save_stats_to_file(self, path):
        if self._best_features is None:
            raise('Model not trained, please fit the model first')
           
        with open(path, 'w') as f:
            f.write('Best score: ' + str(self._best_score))
            f.write('\nBest features: ' + ', '.join(self._best_features))
            f.write('\nFull tree searched: ' + str(self._root._is_subtree_full))
            f.write('\nLongest branch: ' + str(self._longest_tree_branch))
            f.write('\nMetric name: ' + str(self._metric_name))
            f.write('\nScoring function: ' + str(self._scoring_function_name))
            f.write('\nMultiarm strategy: ' + str(self._multiarm_strategy_name))
            f.write('\nEnd strategy: ' + str(self._end_strategy_name))
            f.write('\nCalculations done condition: ' + str(self._calculations_done_condition))
            f.write('\nCalculations budget: ' + str(self._calculations_budget))
            
            f.write('\nScores history: ')
            for sc in self._scores_history:
                f.write('\nscore: ' + str(sc['score']) + ', features: ' + ', '.join(sc['features']))
            
            f.write('\nParameters: ')
            for key, value in self._params.items():
                f.write('\n' + key + ': ' + str(value))
                