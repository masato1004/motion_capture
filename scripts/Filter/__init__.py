from .filter import *

def allowed_filters_name():
    return allowed_filters().keys()

def define_filter(name, **kwargs):
    filters = allowed_filters()
    if name in filters:
        return filters[name](**kwargs)
    else:
        raise ValueError(f"Filter '{name}' not recognized. Available filters: {list(filters.keys())}")