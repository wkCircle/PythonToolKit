import re 
import pandas as pd 
import numpy as np 

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
        ...          'RMSE__sample_weight': None, 
        ...          'MAE__multiple_weight': 'raw_values', 
        ...          'MAE__sample_weight': None, }
        >>> config_parser('RMSE', space)
        {'multiple_weight': 'uniform_average', 'sample_weight': None}
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
    
    output = None 
    if re.search('d$|days?', unit, flags=re.IGNORECASE): 
        output = value * 1
    elif re.search('w$|w-.+|weeks?', unit, flags=re.IGNORECASE): 
        output = value * 7
    elif re.search('m$|months?', unit, flags=re.IGNORECASE): 
        output = value * 30.4368
    elif re.search('Q$|quarters?', unit, flags=re.IGNORECASE): 
        output = value * 91.3106
    else: 
        raise NotImplementedError
    return pd.Timedelta(output, unit='D')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df