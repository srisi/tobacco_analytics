




def check_param_type(variable, expected_type, variable_name=None, function_name=None):
    """
    Checks if the passed variable is the expected type.
    If it is, returns True
    If not, raises an AttributeError

    >>> check_param_type('string', 'test_string', str)
    True
    >>> check_param_type('string', 'not_list', list)
    AttributeError

    :param variable:
    :param variable_name:
    :param expected_type:
    :return:
    """

    if isinstance(variable, expected_type):
        return True
    else:
        if function_name:
            err = f'Attribute Error in {function_name}: '
        else:
            err = 'Attribute Error: '

        if variable_name:
            err += f'{variable_name} should be {expected_type} but a {type(variable)} got passed.'
        else:
            err += f'Expected {expected_type} but a {type(variable_name)} got passed.'

        raise AttributeError(err)


if __name__ == '__main__':
    check_param_type('string', 'test_string', str)

