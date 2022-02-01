from discrete_bottleneck.models.tokenizers import SimpleTokenizer


def is_tokenizer_trained(tokenizer):
    if isinstance(tokenizer, SimpleTokenizer) and not tokenizer.trained:
        return False

    return True
