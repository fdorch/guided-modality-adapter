from typing import List

from transformers import PreTrainedTokenizer


def add_timestamp_tokens(
    tokenizer: PreTrainedTokenizer,
    max_time_s: float = 30.0,
    step_ms: int = 20,
) -> List[str]:
    """
    Adds discrete timestamp tokens <T0.00>, <T0.02>, ... <T30.00>.
    Step is in milliseconds. max_time_s defines maximum audio length.
    """
    new_tokens = []
    t = 0.0
    while t <= max_time_s + 1e-6:
        tok = f"<T{t:.2f}>"
        new_tokens.append(tok)
        t += step_ms / 1000.0

    # add as special tokens
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    return new_tokens


def extend_tokenizer_with_speakers(
    tokenizer: PreTrainedTokenizer,
    num_speakers: int,
    speaker_prefix: str = "<SPK_",
    speaker_suffix: str = ">",
) -> List[str]:
    new_tokens = [
        f"{speaker_prefix}{i:02d}{speaker_suffix}" for i in range(num_speakers)
    ]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    return new_tokens
