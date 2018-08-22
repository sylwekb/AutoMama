class AutoMamaException(Exception):
    pass


class ConfigurationError(Exception):
    pass


class TimeoutError(AutoMamaException):
    pass


class QueryImageNotFound(AutoMamaException):
    pass


class TextNotFound(AutoMamaException):
    pass


class FileDoesNotExist(AutoMamaException, IOError):
    pass


class NotImplementedStrategy(AutoMamaException, NotImplementedError):
    pass
