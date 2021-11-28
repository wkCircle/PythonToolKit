import re 
import pandas as pd 

# warning class  
class DuplicatedValueWarning(UserWarning): 
    """use for category of warning type"""
    pass 

# dict parser that helps extract keyword, this come in handy when using Pipeline() and assinging which configuration belongs to which step of the pipeline.
def config_parser(key: str, space: dict): 
    """
    extract the setting from space dict based on key pattern. 
    Args:
        key (str): form the regex pattern to parse and extract from space dict.
        space (dict): the dict that will be searched. 

    Returns:
        dict: contains the setting related to key pattern, key pattern will be removed from the dict string keys.

    Example: 
        >>> space = {'RMSE__multiple_weight': 'uniform_average', 
                     'RMSE__sample_weight': None, 
                     'MAE__multiple_weight': 'raw_values', 
                     'MAE__sample_weight': None, }
        >>> config_parser('RMSE', space)
        {'multiple_weight': 'uniform_average', 'sample_weight':None}
    """
    # define pattern 
    prefix = f'{key}__\w+'
    # find the keys matching the pattern required 
    keys = re.findall(prefix, '  '.join(space.keys()))
    # modify the keys by deleting the prefix 
    keys_no_prefix = [k.replace(f'{key}__', '') for k in keys]
    return dict(zip(keys_no_prefix, [space[k] for k in keys] ))

def get_equivalent_days(value: float = 1, unit: str = 'D'): 
    unit = unit.lower() 
    if unit is None: 
        return None 
    elif re.search('d$|days?', unit, flags=re.IGNORECASE): 
        output = value * 1
    elif re.search('w$|w-.+|weeks?', unit, flags=re.IGNORECASE): 
        output = value * 7
    elif re.search('m$|months?', unit, flags=re.IGNORECASE): 
        output = value * 30.4368
    elif re.search('Q$|quarters?', unit, flags=re.IGNORECASE): 
        output = value * 91.3106
    print(output, type(output))
    return pd.Timedelta(output, unit='D')