
class TextPredictException(Exception):
    """textpredict core exception.

    Thrown when an error occurs specific to textpredict core concepts.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.message = message
        self.errors = errors


class InvalidLanguageModelException(TextPredictException):
    """Invalid Language Model Exception.

    Thrown when attempting to load a language model from an invalid path"""
    ...


class KenLMInstallationException(TextPredictException):
    """KenLM Installation Exception.

    Thrown when attempting to import kenlm without installing the module"""
    ...
