class NoopLock:

    def __enter__(self):
        # Do nothing
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Do nothing
        pass
