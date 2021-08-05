from typing import List

from .colors import color

def generic_type_template(tipo: str, name: str, behaviour: str, result0: str, result1: str) -> str:
    """Template for feature behaviour reason generated from DICE
    Returns:
         str: generic behaviour based on the type
    """

    dict_type = {
        "category": (f"{name} {behaviour} from {result0} to {result1}."),
        "continuous": (f"{name} {behaviour} from {result0} to {result1}"),
        }

    return dict_type[tipo].format(
            name = name,
            behaviour = behaviour,
            result0 = result0,
            result1 = result1
        )


def get_behaviour(tipo: str, result0: str, result1: str) -> str:
    if tipo == 'category':
        behaviour = 'changes'
    elif tipo == 'continuous':
        if float(result1)-float(result0)>0:
            behaviour = 'increases'
        elif float(result1)-float(result0)<0:
            behaviour = 'decreases'
        else:
            behaviour = 'stays'
    return behaviour

def custom_behaviour_template(tipo: str, feature: str, result0: str, result1: str) -> str:
    """Template for feature behaviour reason generated from DICE
    Returns:
         str: behaviour 
    """

    behaviour = get_behaviour(tipo = tipo, result0 = result0, result1 = result1)
        
    phrase = generic_type_template(tipo =  tipo, name = feature, behaviour = behaviour, result0 = result0, result1 = result1)

    result = f"when {phrase}, "
    
    return result

def custom_model_template(model_type: str, target: str, result0: str, result1: str) -> str:
    """Template for feature behaviour reason generated from DICE
    Returns:
         str: behaviour 
    """

    if model_type == 'classifier':
        tipo = 'category'
    elif model_type == 'regressor':
        tipo = 'continuous'

    behaviour = get_behaviour(tipo = tipo, result0 = result0, result1 = result1)

    phrase = generic_type_template(tipo =  tipo, name = target, behaviour = behaviour, result0 = result0, result1 = result1)

    result = color.BLUE + f" the output of the model {phrase}." + color.END

    return result

def custom_template(tipi: List[str], features: List[str], results0: List[str], results1: List[str], 
                    model_type: str, target: str, target0: str, target1: str) -> str:
    """Template for feature behaviour reason generated from DICE
    Returns:
         str: reason 
    """
    N = len(tipi)
    stringa = ' and '
    Nand = len(stringa)
    feature_phrases = [custom_behaviour_template(tipo = tipi[i], feature = features[i], result0 = results0[i], result1 = results1[i]) for i in range(N)]
    feature_phrase =  ''.join(f'{phrase_feat}{stringa}' for phrase_feat in feature_phrases)
    feature_phrase = feature_phrase[:-Nand]
    model_phrase = custom_model_template(model_type = model_type, target = target, result0 = target0, result1 = target1)

    phrase = feature_phrase+model_phrase
    
    return phrase

    