from dice_ml.reason_generator_interfaces import reason_templates

from dice_ml import Dice

import numpy as np

from typing import List, Dict


class ReasonGeneratorBase:
    def __init__(self, continuous_features: List[str], categorical_features: List[str], outcome: str, model_type: str):
        self.model_type = model_type
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.features = continuous_features+categorical_features
        self.outcome = outcome
        
        self.categorical = 'category'
        self.continuous = 'continuous'
        self.classifier = 'classifier'
        self.regressor = 'regressor'

    def get_types(self, feature_names: List[str]):
        result = []#{self.categorical: [], self.continuous: []}
        for feat in feature_names:
            result += [self.get_type(feat)]
        return result

    def get_type(self, feat: str):
        if feat in self.continuous_features:
            result = self.continuous
        elif feat in self.categorical_features:
            result = self.categorical
        elif feat == self.outcome:
            result = self.categorical if self.model_type == self.classifier else self.continuous
        return result
    
    def access_dict(self, dictionary: Dict, name: str, key: str):
        result = self.format(dictionary[name][key], self.get_type(name))
        return result
    
    def format(self, stringa: str, tipo: str):
        if tipo == self.categorical:
            result = f'{int(float(stringa)):.0f}'
        elif tipo == self.continuous:
            result =  f'{stringa:.3f}'
        return result

    def get_results_lists(self, dictionary: Dict, key0: str = 'result0', key1: str = 'result1'):
        results0 = []
        results1 = []
        dictionary_copy = dictionary.copy()
        del dictionary_copy[self.outcome]
        for feat, vals in dictionary_copy.items():
            tipo = self.get_type(feat)
            results0 += [self.format(vals[key0], tipo)]
            results1 += [self.format(vals[key1], tipo)]
            
        return results0, results1

    def order_top_features(self, top_features_per_instance: Dict, n: int):
        sorted_dict = dict(sorted(top_features_per_instance.items(), key = lambda item: item[1]))
        keys = list(sorted_dict.keys())[:n]
        values = list(sorted_dict.values())[:n]
        return keys, values

    
    def generate_reasons(self, cf_examples_list: List, top_features_per_instance_list: List[Dict], n_important: int = 10, result0: str = 'result0', result1: str = 'result1',
                        target0: str = 'target0', target1: str = 'target1', verbose = True):
        for i, cf_example in enumerate(cf_examples_list):
            if verbose:
                print(f'For query number {i}:')
            keys_important, _ = self.order_top_features(top_features_per_instance_list[i], n_important)
            print(f'Most important features: {keys_important}')
            changes_list = self.check_changes(cf_example)
            for j, dictionary in enumerate(changes_list):
                if verbose:
                    print(f'Counterfactual number {j} of query {i}:')
                features = list(dictionary.keys())
                features.remove(self.outcome)
                feature_types = self.get_types(features)
                results0, results1 = self.get_results_lists(dictionary, key0 = result0, key1 = result1)
                
                target0str = self.access_dict(dictionary, self.outcome, target0)
                target1str = self.access_dict(dictionary, self.outcome, target1)
                
                reason = reason_templates.custom_template(tipi = feature_types, features = features, results0 = results0, results1 = results1, 
                        model_type = self.model_type, target = self.beautify(self.outcome), target0 = target0str, target1 = target1str)
                print(reason)
            print('\n')

    def beautify(self, stringa, clothing = 'square brackets'):
        if clothing == 'square brackets':
            result = '['+stringa+']'
        return result
        
    def check_changes(self, cf_example: List) -> List:
        '''
        List of the type:
            new_cfs_changes[index][col] = {'result0': old, 'result1': new}
        '''
        pass
        


class ReasonGenerator(ReasonGeneratorBase):
    def __init__(self, exp: Dice):
        super().__init__(continuous_features = exp.data_interface.continuous_feature_names, 
                         categorical_features = exp.data_interface.categorical_feature_names,
                         outcome = exp.data_interface.outcome_name,
                         model_type = exp.model.model_type)
        
    def check_changes(self, cf_example, atol = 0.1) -> List:
        org_instance = cf_example.test_instance_df
        
        if cf_example.final_cfs_df_sparse is not None:
            new_cfs = cf_example.final_cfs_df_sparse
        else:
            new_cfs = cf_example.final_cfs_df
            
        new_cfs_changes = []
        
        for index, row in new_cfs.iterrows():
            
            new_cfs_changes += [{self.outcome: {'target0': org_instance[self.outcome].iloc[0], 'target1': row[self.outcome]}}]
            
            for col in self.features:
                old = org_instance[col].iloc[0]
                new = row[col]
                
                add = False
                if col in self.continuous_features:
                    if not np.isclose(old, new, atol = atol):
                        add = True
                elif col in self.categorical_features:
                    if old != new:
                        add = True
                        
                if add:
                    new_cfs_changes[index][col] = {'result0': old, 'result1': new}
                
        return new_cfs_changes