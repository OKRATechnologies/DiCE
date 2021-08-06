from dice_ml.reason_generator_interfaces import reason_templates

from dice_ml import Dice

import numpy as np

from typing import List, Dict

from .colors import color

import re

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
            try:
                result = f'{int(float(stringa)):.0f}'
            except:
                result = stringa
        elif tipo == self.continuous:
            result =  f'{stringa:.3f}'
        return result

    def get_results_lists(self, dictionary: Dict, key0: str = 'result0', key1: str = 'result1', features_to_include: List = None):
        if features_to_include is None:
            features_to_include = self.features

        results0 = []
        results1 = []
        dictionary_copy = dictionary.copy()
        del dictionary_copy[self.outcome]
        for feat in features_to_include:
            vals = dictionary_copy[feat]
            tipo = self.get_type(feat)
            results0 += [self.format(vals[key0], tipo)]
            results1 += [self.format(vals[key1], tipo)]
            
        return results0, results1

    def order_top_features(self, top_features_per_instance: Dict, threshold_importance: float):
        new = {}
        for k, v in top_features_per_instance.items():
            if v > threshold_importance:
                new[k] = v
        sorted_dict = dict(sorted(new.items(), key = lambda item: item[1]))
        keys = list(sorted_dict.keys())
        values = list(sorted_dict.values())
        return keys, values

    
    def generate_reasons(self, cf_examples_list: List, top_features_per_instance_list: List[Dict], threshold_importance: float = 0.3, result0: str = 'result0', result1: str = 'result1',
                        target0: str = 'target0', target1: str = 'target1', verbose = True):

        all_reasons = []

        for i, cf_example in enumerate(cf_examples_list):
            all_reasons += [[]]
            if verbose:
                print(color.RED + color.BOLD + f'For query number {i}:' + color.END, '\n')
            keys_important, _ = self.order_top_features(top_features_per_instance_list[i], threshold_importance)
            if len(keys_important)>0:
                print(f'Most important features with threshold of {threshold_importance}: {keys_important}')
            else:
                print(f'No feature with threshold of at least {threshold_importance}')

            print('\n')

            changes_list = self.check_changes(cf_example)
            for j, dictionary in enumerate(changes_list):
                if verbose:
                    print(color.BOLD + f'Counterfactual number {j} of query {i}:' + color.END)
                #If no local importance above threshold, just show all of the changes
                features = list(set(list(dictionary.keys()) ) & set(keys_important)) if len(keys_important)>0 else list(dictionary.keys())

                if self.outcome in features:
                    features.remove(self.outcome)
                    
                feature_types = self.get_types(features)
                results0, results1 = self.get_results_lists(dictionary, key0 = result0, key1 = result1, features_to_include = features)
                
                target0str = self.access_dict(dictionary, self.outcome, target0)
                target1str = self.access_dict(dictionary, self.outcome, target1)

                reason = reason_templates.custom_template(tipi = feature_types, features = features, results0 = results0, results1 = results1, 
                        model_type = self.model_type, target = self.beautify(self.outcome), target0 = target0str, target1 = target1str)
                print(reason, '\n')
                all_reasons[i] += [reason]
            print('\n')

        return all_reasons

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


    def modify(self, s):
        return re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), s)
        


class ReasonGenerator(ReasonGeneratorBase):
    def __init__(self, exp: Dice):
        super().__init__(continuous_features = exp.data_interface.continuous_feature_names, 
                         categorical_features = exp.data_interface.categorical_feature_names,
                         outcome = exp.data_interface.outcome_name,
                         model_type = exp.model.model_type)
        
    def check_changes(self, cf_example, atol = 1e-8) -> List:
        org_instance = cf_example.test_instance_df
        
        #if cf_example.final_cfs_df_sparse is not None:
        #    new_cfs = cf_example.final_cfs_df_sparse
        #else:
        new_cfs = cf_example.final_cfs_df
            
        new_cfs_changes = []
        
        i = 0

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
                    new_cfs_changes[i][col] = {'result0': old, 'result1': new}
                
            i += 1

        return new_cfs_changes