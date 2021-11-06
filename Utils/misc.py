import re 

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
