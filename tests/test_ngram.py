from ngram import NGramLanguageModel

SYMBOL_SET_LOWER = list("abcdefghijklmnopqrstuvwxyz")

# Test trying to construct using a filename that doesn't exist
def test_constructor_bogus_model(capsys):
    lm = NGramLanguageModel(symbol_set=SYMBOL_SET_LOWER,
                               lm_path="bogus.arpa")
    assert lm is not None
    assert lm.model is None
    captured = capsys.readouterr()
    assert captured.out.startswith("ERROR:")
