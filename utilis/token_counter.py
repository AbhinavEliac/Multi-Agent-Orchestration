import threading

_local = threading.local()

def reset_token_counter():
    _local.prompt_tokens = 0
    _local.completion_tokens = 0
    _local.total_tokens = 0

def add_tokens(prompt: int, completion: int):
    if not hasattr(_local, "prompt_tokens"):
        reset_token_counter()
    _local.prompt_tokens += prompt
    _local.completion_tokens += completion
    _local.total_tokens += (prompt + completion)

def get_tokens():
    if not hasattr(_local, "prompt_tokens"):
        return 0, 0, 0
    return _local.prompt_tokens, _local.completion_tokens, _local.total_tokens
