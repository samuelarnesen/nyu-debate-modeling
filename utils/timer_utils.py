from utils.logger_utils import LoggerUtils

from functools import wraps
import time


def timer(custom_name: str = None):
    """Decorator to time a function"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            name = custom_name if custom_name else func.__name__
            LoggerUtils.get_default_logger(__name__).debug(f"{name} completed in {elapsed_time:.1f} seconds")
            return result

        return wrapper

    return decorator
