import functools

def traceable(func=None, **decorator_kwargs):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator
