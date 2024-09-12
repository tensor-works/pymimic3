import functools
from utils.IO import *


class repeat:

    def __init__(self, times):
        self.times = times

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(self.times):
                func(*args, **kwargs)

        return wrapper


class retry:
    """
    A decorator class that retries a test function a specified number of times if it raises an exception.

    Parameters
    ----------
    retries : int
        The number of times to retry the function if it fails.
    """

    def __init__(self, retries: int):
        self.retries = retries

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < self.retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    tests_io(f"Attempt {attempts} failed: {e}")
                    if attempts == self.retries:
                        tests_io("Max retries reached. Test failed.")
                        raise
                    tests_io(f"Retrying... ({attempts}/{self.retries})")

        return wrapper
