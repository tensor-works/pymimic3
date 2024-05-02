import pytest
import functools


class repeat:

    def __init__(self, times):
        self.times = times

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(self.times):
                func(*args, **kwargs)

        return wrapper
