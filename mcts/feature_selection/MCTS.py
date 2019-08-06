from mcts.feature_selection.Node import *
from mcts.feature_selection.DefaultSettings import *
from mcts.feature_selection.CV import *
from mcts.feature_selection.Preprocessing import *
from mcts.feature_selection.MultiArmStrategies import *
from mcts.feature_selection.EndStrategies import *
from mcts.feature_selection.ScoringFunctions import *
from mcts.feature_selection.BuildInMetrics import *
from mcts.feature_selection.GlobalScores import *
from mcts.feature_selection.NodeAdder import *
from mcts.feature_selection.DrawTree import draw_tree
import time
from sklearn.ensemble import RandomForestClassifier

class MCTS:
    """Class for MCTS"""
    def __init__(self, 
                 model,
                 calculations_done_condition = 'iterations',
                 calculations_budget = 10,
                 params = None,
                 metric = 'roc_auc', 
                 scoring_function = 'UCB1_rave', 
                 multiarm_strategy = 'discrete', 
                 end_strategy = 'default',
                 preprocess = True):
        
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
        
        if params is None:
            self._params = DefaultSettings.get_default_params()
        else:
            self._params = DefaultSettings.merge_params(params)
        
    def fit(self, data, out_variable, pos_class = 'numeric', warm_start = True,
                 calculations_done_conditions = None,
                 calculations_budget = None):
        
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
        
        score = CV.cv(self._metric, self._metric_name, self._model, data[list(node._features)], 
                      out_variable, self._params['cv'])
        self._update_nodes(used_nodes, score)
        self._global_scores.update_score(node._features, score)
        
        if(score > self._best_score):
            self._best_score = score
            self._best_features = list(node._features)
            self._scores_history = self._scores_history.append({
                'score': score, 
                'features': self._best_features,
                'time': time.time() - self._time,
                'iteration': self._iterations
            },ignore_index=True)
        
        if(self._longest_tree_branch < used_nodes_index):
            self._longest_tree_branch = used_nodes_index

    
    def _update_nodes(self, used_nodes, score):
        for i in range(len(used_nodes)):
            if used_nodes[i] is None:
                return
            used_nodes[i].add_score(score)
    
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
        if self._iterations % 20 == 0:
            print('Iterations done: ' + str(self._iterations))
            
        if self._calculations_done_condition == 'iterations':
            return self._iterations > self._calculations_budget 
        else:
            return (time.time() - self._time) > self._calculations_budget 
    
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
            
            f.write('\nScores history: ')
            
            f.write('\nParameters: ')
            for key, value in self._params.items():
                f.write('\n' + key + ': ' + str(value))
                
        self.get_search_history().to_csv(path + '.csv')
        self._global_scores.get_g_rave_dataframe().to_csv(path + '_g_rave.csv')
        self._global_scores.get_l_rave_dataframe().to_csv(path + '_l_rave.csv')