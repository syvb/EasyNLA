"""AV prompt construction shared by the RL trainers (and eval/probe scripts)."""


def build_prompt_text(prompt_msgs, inject_char, tokenizer, *, enable_thinking: bool | None = None):
    """Apply the chat template; substitute the <INJECT> placeholder with inject_char.

    When `enable_thinking` is None (default), omits the kwarg for full backward
    compatibility. When explicitly set (e.g. False for Qwen3 cold-start AV runs),
    passes `enable_thinking` to `apply_chat_template` if supported by the tokenizer.
    """
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(msgs, **kwargs)
