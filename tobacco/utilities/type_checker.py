


from typing import Union





def check_param_type(variable, expected_type, variable_name=None, function_name=None):
    """
    Checks if the passed variable is the expected type.
    If it is, returns True
    If not, raises an AttributeError

    >>> check_param_type('string', str, 'test_string', 'test_func')
    True
    >>> check_param_type('string', list, 'test_string', 'other_func')
    Traceback (most recent call last):
    ...
    AttributeError: In other_func: test_string should be <class 'list'> but a <class 'str'> got passed.

    # Unions work as well
    >>> check_param_type('string', Union[list, str], 'test_string', 'union_func')
    True

    :param variable:
    :param variable_name: type or Union
    :param expected_type:
    :return:
    """

    try:
        if isinstance(variable, expected_type):
            return True
    except TypeError:
        # checking if the expected type is a Union is... weird. One way is to look for the
        # __args__list.
        if hasattr(expected_type, '__args__'):
            for t in expected_type.__args__:
                if isinstance(variable, t):
                    return True
        else:
            raise NotImplementedError("Check param expects a type or Union. You presumably passed"
                                      f" neither. Var: {variable}. Type: {type(variable)}. "
                                      f"Expected type: {expected_type}.")

    if function_name:
        err = f'In {function_name}: '
    else:
        err = ''

    if variable_name:
        err += f'{variable_name} should be {expected_type} but a {type(variable)} got passed.'
    else:
        err += f'Expected {expected_type} but a {type(variable_name)} got passed.'

    raise AttributeError(err)


if __name__ == '__main__':
    check_param_type('string', list, 'test_string', 'other_func')

