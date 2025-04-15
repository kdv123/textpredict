
class TextSlingerException(Exception):
    """textslinger core exception.

    Thrown when an error occurs specific to textslinger core concepts.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.message = message
        self.errors = errors


class InvalidLanguageModelException(TextSlingerException):
    """Invalid Language Model Exception.

    Thrown when attempting to load a language model from an invalid path"""
    ...


class KenLMInstallationException(TextSlingerException):
    """KenLM Installation Exception.

    Thrown when attempting to import kenlm without installing the module"""
    ...
