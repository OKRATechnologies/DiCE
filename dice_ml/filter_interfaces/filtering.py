from dice_ml.counterfactual_explanations import CounterfactualExplanations


class FilterChanges:
    def __init__(self, tabu_changes):
        self.tabu_changes = tabu_changes
        
        self.coupled_features = []
        for k in tabu_changes.keys():
            if type(k) is tuple:
                self.coupled_features += [k]
    
    def is_feasible_category(self, feature, change, tabu_changes):
        if feature not in tabu_changes.keys():
            return True
        else:
            if change in tabu_changes[feature]:
                return False
            else:
                return True
            
    def is_feasible_continuous(self, feature, change, tabu_changes):
        if feature not in tabu_changes.keys():
            return True
        else:
            if tabu_changes[feature][0](change):
                return False
            else:
                return True
            
    def is_feasible(self, feature, change, continuous = False, tabu_changes = None):
        if tabu_changes is None:
            tabu_changes = self.tabu_changes
        if continuous:
            result = self.is_feasible_continuous(feature, change, tabu_changes)
        else:
            result = self.is_feasible_category(feature, change, tabu_changes)
        return result
    
    def is_feasible_couple(self, feature_couple, change_couple, cont_couple):
        #I already know feature_couple is in tabu_changes
        
        tabu_change_couple = self.tabu_changes[feature_couple]
        
        is_feasible = False
        
        #should work also for non couple in principle
        for i, c in enumerate(zip(feature_couple, change_couple)):
            feature, change = c
            tabu_change = {feature: tabu_change_couple[i]}
            cont = cont_couple[i]
            is_i = self.is_feasible(feature, change, cont, tabu_change)
            is_feasible = is_feasible or is_i 
            
        return bool(is_feasible)
    
    
    def _get_change_cont(self, o, n, tipo):
        if tipo == 'category' or tipo == 'object':
                temp = [o, n]
                continuous = False
        else:
            temp = n-o
            continuous = True
        return temp, continuous
    
    
    def filter_changes_single(self, changes):
        
        doable = True
        
        #Single features only
        for f, v in changes.items():
            o, n, tipo = v
            temp, continuous = self._get_change_cont(o, n, tipo)

            if not self.is_feasible(f, temp, continuous, self.tabu_changes):
                doable = False
                break
        
        return doable
    
    
    def filter_changes_couple(self, changes):
        
        doable = True
        
        for k in self.coupled_features:
            f1, f2 = k
            keys = changes.keys()
            if (f1 in keys) and (f2 in keys):
                o, n, tipo = changes[f1]
                diff1, continuous1 = self._get_change_cont(o, n, tipo)
                
                o, n, tipo = changes[f2]
                diff2, continuous2 = self._get_change_cont(o, n, tipo)
                
                if not self.is_feasible_couple((f1, f2), (diff1, diff2), (continuous1, continuous2)):
                    doable = False
                    break
        
        return doable
        
            
    def filter_changes(self, changes):
        
        #Single features only
        doable1 = self.filter_changes_single(changes)
        if not doable1:
            return doable1
                
        #Coupled features
        doable2 = self.filter_changes_couple(changes)

        doable = doable1 and doable2
            
        return doable

    def filtered_cf_list(self, cf_examples_list):

        #here it assumes you have the same number of CFs for each example
        total_CFs = cf_examples_list[0].final_cfs_df.shape[0] 
        
        
        lista = self.changed_from_counterfactual(cf_examples_list)
        
        ok_indices = {}
        min_number_ok_cfs = 1e6 #absurd high number

        #Here I make sure that all the cfs for a given example are feasible
        #Ideally you want to generate losts of cfs for each example, and take the minimum number of CFs to retain all the examples
        #You could also elimante each example with a non feasible CFs, but this does not make lots of sense
        for i, element in enumerate(lista):
            ok_indices[i] = []
            cf_examples_list[i].final_cfs_df.reset_index(drop = True, inplace = True)
            for j, el in enumerate(element):
                if self.filter_changes(el):#, continuous = el in self.continuous):
                    ok_indices[i] += [j]
                else:
                    cf_examples_list[i].final_cfs_df.drop(j, axis = 0, inplace = True)
            cf_examples_list[i].final_cfs_df.reset_index(inplace = True, drop = True)
                    
            min_number_ok_cfs = min(min_number_ok_cfs, len(ok_indices[i]))

        excluded = total_CFs-min_number_ok_cfs

        print(f'Number of excluded counterfactuals is {excluded}')
            
        for i, _ in enumerate(lista):
            cf_examples_list[i].final_cfs_df = cf_examples_list[i].final_cfs_df.truncate(after = min_number_ok_cfs-1, axis = 0)
        
        return  cf_examples_list

    def changed_from_counterfactual(self, cf_examples_list):
        
        results = []
        for example in cf_examples_list:
            original = example.test_instance_df.values.tolist()[0]
            features = example.test_instance_df.columns.tolist()
            types = example.test_instance_df.dtypes
            inner_list = []
            for new in example.final_cfs_df.values.tolist():
                result = self.compare_lists(features, types, original, new)
                #if self.outcome in result.keys():
                #    del result[self.outcome]
                inner_list += [result]
            
            results += [inner_list]
        
        return results

    def compare_lists(self, features, types, original, new):
        '''
        Takes original instance in the form of list, new instance, and compare.

        Returns a dictionary with changed features. 

        The dictionary is in the form of changed[key] = (original_value, new_value, type of feature)

        '''

        changed = {}
        for f, o, n in zip(features, original, new):
            if o != n:
                changed[f] = (o, n, types[f].name)
        return changed
    
