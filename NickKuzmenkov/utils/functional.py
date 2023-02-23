import inspect

"""Argument parsing"""


def list_global_variables(condition=None, filter_underscore=True, **kwargs):
    def cond(k):
        if k == "list_global_variables":
            return False
        if filter_underscore and k.startswith('_'):
            return False
        if condition is not None:
            return condition(k)
        return True

    return list(filter(cond, globals().keys()))


def list_global_constants(condition=None, filter_underscore=True, **kwargs):
    upper = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890')

    def cond(k):
        if condition is not None and not condition(k):
            return False
        if not all(c in upper for c in k):
            return False
        return True

    return list_global_variables(condition=cond, filter_underscore=filter_underscore, **kwargs)


def list_valid_args(func):
    return list(inspect.signature(func).parameters.keys())


def retrieve_global_variables(keys):
    return {k: eval(k) for k in keys}


def retrieve_global_valid_constants(func):
    valid_args = list_valid_args(func)
    constants = list_global_constants(condition=lambda k: not callable(eval(k)))
    keys = [k for k in constants if k.lower() in valid_args]
    return retrieve_global_variables(keys)


def lower_keys(d):
    return {k.lower(): v for k, v in d.items()}