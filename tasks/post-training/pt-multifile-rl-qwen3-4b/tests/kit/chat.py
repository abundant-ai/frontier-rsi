"""Chat specifics, in one place, for any chat model with a chat template (tested: Qwen2.5 / Qwen3, Llama 3.1 / 3.2,
Gemma 3, Ministral 3, Granite 4.2 instruct models).

Prompts are rendered with the tokenizer's own chat template, thinking off where the template has a switch (Qwen3 and
Granite 4 read `enable_thinking`; the generation prompt then opens the reply with an empty think block). A turn ends
when the model emits the template's end-of-turn token (<|im_end|>, <|eot_id|>, <end_of_turn>, </s>) or one of its eos
tokens (tokenizer and generation_config.json). For multi-turn episodes, an environment's reply is appended as a new
user turn with turn_suffix_ids(): the tokens the template puts between the model's reply and the next reply, taken
from two renders of a placeholder conversation, so earlier turns are never re-encoded.
"""
import functools
import re

IM_END = "<|im_end|>"   # Qwen's end of turn: the fallback when the template cannot be used for a suffix
THINKING_SWITCHES = ("enable_thinking", "thinking")   # Qwen3 / Granite 4, Granite 3
END_OF_TURN = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<|end_of_text|>")
REPLY = "xyzzy"   # a placeholder reply that no template or system prompt contains


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def template_kwargs(tok, thinking=False):
    """The thinking switch(es) this tokenizer's chat template reads, set to `thinking`; {} if it has none."""
    t = str(getattr(tok, "chat_template", None) or "")
    return {k: thinking for k in THINKING_SWITCHES if re.search(rf"\b{k}\b", t)}


def render(tok, messages, thinking=False, generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt,
                                   **template_kwargs(tok, thinking))


def prompt_ids(tok, messages, thinking=False):
    return encode(tok, render(tok, messages, thinking))


def _after(tok, text, anchor):
    """Token ids of `text` after the first `anchor`, cut where the tokens of everything up to it end (ValueError if a
    token straddles that point)."""
    end = text.index(anchor) + len(anchor)
    full, head = encode(tok, text), encode(tok, text[:end])
    if full[:len(head)] != head:
        raise ValueError("no token boundary after the reply")
    return full[len(head):]


def _closing(tok, thinking=False):
    """What the template writes after an assistant reply, as token ids (e.g. <|im_end|> \\n)."""
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": REPLY}]
    return _after(tok, render(tok, msgs, thinking, generation_prompt=False), REPLY)


@functools.lru_cache(maxsize=None)
def _config_eos(path):
    try:
        from transformers import GenerationConfig
        e = GenerationConfig.from_pretrained(path).eos_token_id
    except Exception:   # no generation_config.json (e.g. a bare tokenizer dir)
        return frozenset()
    return frozenset(e if isinstance(e, (list, tuple)) else [e])


def generation_eos(tok):
    """The tokenizer's eos id plus every eos id in the model's generation_config.json (int or list)."""
    return ({tok.eos_token_id} | _config_eos(getattr(tok, "name_or_path", None))) - {None}


def end_of_turn_id(tok):
    """The token the chat template closes an assistant turn with, or None."""
    try:
        close = _closing(tok)
        if close:
            return close[0]
    except Exception:
        pass
    t = str(getattr(tok, "chat_template", None) or "")
    return next((tok.convert_tokens_to_ids(c) for c in END_OF_TURN if c in t), None)


def stop_ids(tok):
    return sorted((generation_eos(tok) | {end_of_turn_id(tok)}) - {None})


def reply_end_id(tok):
    """The token an SFT target ends with: the end-of-turn token if generation stops on it by default (it is an eos of
    the tokenizer or generation config: instruct models), else the eos (e.g. a Qwen3 base model's <|endoftext|>)."""
    eot = end_of_turn_id(tok)
    return eot if eot in generation_eos(tok) else tok.eos_token_id


def final_text(text):
    """The part of a reply that counts: after the last </think>, or nothing if a <think> was left open."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return "" if "<think>" in text else text.strip()


def turn_suffix(observation, ended_with_im_end, thinking=False):
    """Qwen's text between turns, closing the model's turn (unless it already emitted <|im_end|>) and opening the
    next one: turn_suffix_ids() falls back to it when the template cannot give a suffix."""
    head = "" if ended_with_im_end else IM_END
    return (f"{head}\n<|im_start|>user\n{observation}{IM_END}\n<|im_start|>assistant\n"
            + ("" if thinking else "<think>\n\n</think>\n\n"))


def turn_suffix_ids(tok, observation, last_id=None, thinking=False):
    """Token ids to append after the model's reply (whose last token is last_id) so the observation is the next user
    turn and the model's next reply follows: the tokens after the reply in a render with the next user turn, minus the
    template's closing token if the model already emitted it."""
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": REPLY}]
    try:
        close = _closing(tok, thinking)
        ids = _after(tok, render(tok, msgs + [{"role": "user", "content": observation}], thinking), REPLY)
        if not close or ids[:len(close)] != close:
            raise ValueError("the closing of a reply changes with the next turn")
    except Exception:
        im_end = tok.convert_tokens_to_ids(IM_END)
        return encode(tok, turn_suffix(observation, last_id is not None and last_id == im_end, thinking))
    return ids[1:] if last_id is not None and last_id == ids[0] else ids


def history_ids(tok, messages, thinking=False):
    """The prompt ids for the model's next turn after a multi-turn exchange (user, assistant, user, ..., user), built
    the way rollouts/multiturn.py builds an episode: each earlier reply ends with reply_end_id(), then turn_suffix_ids()."""
    ids = prompt_ids(tok, messages[:1], thinking)
    end = reply_end_id(tok)
    for k in range(1, len(messages) - 1, 2):
        ids += encode(tok, messages[k]["content"].strip()) + [end] + turn_suffix_ids(tok, messages[k + 1]["content"], end,
                                                                                    thinking)
    return ids


def render_history(tok, messages, thinking=False):
    """history_ids() as text (for reading; train on the ids)."""
    return tok.decode(history_ids(tok, messages, thinking))


def model_class(path):
    """The transformers class config.json names (what TRL's GRPOTrainer loads), so every script loads, trains and
    saves the same class with the base's tensor names: Gemma 3 and Ministral 3 instruct checkpoints are multimodal
    (Ministral 3 does not load with AutoModelForCausalLM at all). AutoModelForCausalLM if it names no known class."""
    import transformers
    from transformers import AutoConfig
    arch = (AutoConfig.from_pretrained(path).architectures or [None])[0]
    return getattr(transformers, arch or "", None) or transformers.AutoModelForCausalLM


def lora_exclude(path):
    """peft exclude_modules for a LoRA on `path`: a multimodal checkpoint's (Gemma 3, Gemma 4, Ministral 3) vision and
    audio towers and their projectors, whose projections share the language model's names (q_proj, ...); None for a
    text-only model."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    return r".*(vision_tower|vision_model|multi_modal_projector|audio_tower|embed_vision|embed_audio).*" if hasattr(cfg, "vision_config") else None


def attn_implementation(path=None, choice="auto"):
    """The attention kernel to load `path` with for training (accel.attn_implementation: FlashAttention 3 on Hopper,
    FlashAttention 2 on other sm80+ GPUs, else SDPA; `choice` is --attn auto|sdpa|fa2|fa3), as a from_pretrained
    attn_implementation string."""
    import accel
    return accel.attn_implementation(path, choice)[0]
