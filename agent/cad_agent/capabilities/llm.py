"""Transport to vLLM: build sampling parameters and send the request.

The layer is deliberately dumb: it knows nothing about candidates or prompts, only
how an OpenAI-compatible endpoint works. Everything meaningful (what to show the
model, how to interpret the answer) lives higher up: step generation in ``propose``,
calls to the decision agent in ``resources.ask_agent`` (which also holds the call and
context caps), prompt texts in the policy itself.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)

# Retries are on by default, deliberately: in a run of a thousand parts a single
# dropped connection must not cost a part. The attempt count goes into the technical
# metrics, so real cost hidden behind "retries" stays visible.
#
# Not ALL errors are retried: see `is_permanent_error`. A retry cures a dropped
# connection and server overload, but a request rejected on its content (prompt too
# long, bad parameter) will be rejected just the same the second and third time,
# while the run pays for the sleep between attempts.
DEFAULT_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 1.0

# Prompt size estimate without a tokenizer. The run may live in an environment
# without the assistant's tokenizer at all (the weights are on the server), and
# asking `/tokenize` on every call is an extra round trip for a decision that must
# be made before sending.
#
# Characters per token is chosen to err toward "more tokens than it looks": the
# selection prompt is code with dense coordinate lists (`(-12.34,56.78)`), where
# the tokenizer gives ~2.5-3 characters per token rather than the ~4 usual for prose.
# Underestimating is more dangerous than overestimating: an overestimate costs an
# unnecessary fallback to a prompt without code, an underestimate costs a 400 from
# the server, i.e. the same fallback but after a round trip.
CHARS_PER_TOKEN = 3.0

# Tokens reserved per image. The exact number is set by the SERVER's
# `mm-processor-kwargs` (`min_pixels`/`max_pixels`), which the run does not see: the
# `launch` section is read only by `run_system.sh`. The reserve is generous, because
# being wrong here means a 400 on the very call we are checking.
# Overridable via `experiment.agent.image_tokens`.
AGENT_IMAGE_TOKENS = 4096

# Margin below the cap: system prompt, chat markup, service tokens of the
# multimodal block. None of it is in the estimate; it costs tens of tokens.
CONTEXT_SAFETY_MARGIN = 256



# The exception lives in the seam (`harness/search_types.py`) but is raised here:
# the policy must catch it and must not know about `openai`. The name is re-exported
# for callers: `llm_mod.PromptTooLarge` keeps working.
from cad_agent.harness.search_types import (  # noqa: E402,F401
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    IMAGE_SLOT,
    PromptTooLarge,
)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def is_permanent_error(exc: BaseException) -> bool:
    """Whether repeating this request would give exactly the same answer.

    Anything the server rejected on the request's content (4xx), except two cases:
    429 means "busy right now" and 408 is a timeout, and both are cured by a retry.
    """
    code = _status_code(exc)
    if code is None:
        return False
    return 400 <= code < 500 and code not in (408, 429)


def is_context_length_error(exc: BaseException) -> bool:
    """The request was rejected specifically because of prompt length, not for another reason.

    Recognized by text and by parameter name: for vLLM it is a 400 with
    `parameter=input_tokens` and a message about the maximum context length, for
    OpenAI the code `context_length_exceeded`. One matching marker is enough: a false
    positive costs a question to the agent without program code, a miss costs a
    silent fallback to metric-based selection.
    """
    if _status_code(exc) != 400:
        return False
    text = str(exc).lower()
    markers = (
        "maximum context length",
        "context_length_exceeded",
        "input_tokens",
        "longer than the maximum",
        "reduce the length",
    )
    return any(marker in text for marker in markers)


def estimate_prompt_tokens(
    text: str,
    images: int = 0,
    image_tokens: int = AGENT_IMAGE_TOKENS,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> int:
    """Estimate the prompt size in tokens before sending, without a tokenizer.

    `images` is their COUNT, not a presence flag: a question may carry a list of
    images, and a guard that counted one would let through a prompt several times
    longer than estimated, so the pre-send check would guard nothing.
    """
    per_token = float(chars_per_token) if chars_per_token > 0 else CHARS_PER_TOKEN
    tokens = int(len(text or "") / per_token) + 1
    return tokens + int(image_tokens) * max(0, int(images))


def fetch_context_limit(client: OpenAI, model_name: str) -> int | None:
    """Ask the server for the model's context length (`/v1/models`).

    The only source of truth available to the run: the cap is set by the
    `--max-model-len` flag in the `launch` section, which only `run_system.sh` reads.
    Duplicating the number as a second config key would create exactly the kind of
    divergence that the rule "servers and the run take addresses from one place"
    exists to prevent.

    Returns `None` if the server did not report `max_model_len` (a vLLM extension,
    absent from the plain OpenAI API). Then the cap is unknown, the length check is
    disabled, and the safety net is the server's 400.
    """
    try:
        listing = client.models.list()
    except Exception:
        logger.warning("Could not query /v1/models for the context length of model %s", model_name, exc_info=True)
        return None

    entries = list(getattr(listing, "data", None) or [])
    chosen = None
    for entry in entries:
        if getattr(entry, "id", None) == model_name:
            chosen = entry
            break
    if chosen is None and entries:
        chosen = entries[0]
    if chosen is None:
        return None

    value = getattr(chosen, "max_model_len", None)
    if value is None:
        # vLLM puts extensions both into `model_extra` (pydantic) and into a plain
        # dict, depending on the client version.
        extra = getattr(chosen, "model_extra", None) or {}
        value = extra.get("max_model_len")
    if value is None and isinstance(chosen, dict):
        value = chosen.get("max_model_len")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class LLMCall:
    """Result of a call to the endpoint together with its cost.

    Cost is measured in calls and tokens, and visual calls are counted separately:
    an image costs many times more than text, and a single counter would make the
    visual channel free.
    """

    text: str = ""
    model: str = ""
    has_image: bool = False
    latency_sec: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 1
    error: str | None = None
    errors: list[str] = field(default_factory=list)
    # An error a retry does not cure (see `is_permanent_error`). Kept apart from
    # the ordinary one: `tech.json` should show that the call failed once and for
    # a definite reason, not that retries "somehow did not help".
    permanent: bool = False
    # The prompt did not fit the context. Counted apart from other failures: it is
    # not an endpoint fault but too long a question, and is cured by shortening.
    context_overflow: bool = False
    # All answers of the call. With `n=1` this is `[text]`; with `n>1`, k samples
    # obtained by **one** request. The image is then encoded, sent and
    # preprocessed once, not k times.
    texts: list[str] = field(default_factory=list)
    # How many answers were requested. Separate from `len(texts)`: the endpoint may
    # return fewer, and the difference is visible only if both numbers are stored.
    n_requested: int = 1
    # Reasoning of a thinking model, one field per answer. Filled only when the
    # server runs with `--reasoning-parser`: it extracts the block from the text
    # itself and puts it into `message.reasoning_content`; without this field the
    # reasoning reached NOWHERE, neither the prompt (where it is not needed) nor the
    # run log (where it is: the dialogue must be reconstructible from the part's
    # directory). Without the parser the block stays inside `text` and this is
    # empty; whoever writes the log tells the two forms apart.
    reasonings: list[str] = field(default_factory=list)
    reasoning: str = ""
    # How the endpoint ended each answer. `length` means "hit `max_tokens`", i.e. the
    # answer was CUT OFF, not finished. The field exists because the text cannot
    # tell: a thinking model spends the cap on reasoning, the cut falls in its
    # middle, and a text without the closing `</think>` looks exactly like an
    # ordinary answer the parser just did not like. Ask the endpoint, not the text.
    finish_reasons: list[str] = field(default_factory=list)
    finish_reason: str = ""
    # Function calls parsed by the SERVER (`--enable-auto-tool-choice
    # --tool-call-parser`), per answer: `{"id", "name", "arguments"}`, where
    # `arguments` is a JSON string as the endpoint returns it. Empty when no tools
    # were passed or the model did not call any: then everything it said is in `text`.
    tool_calls_all: list[list[dict[str, str]]] = field(default_factory=list)
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    # How many images went in one call. `has_image` answers a different question
    # (whether the call is visual or text, which drives the budget counter), and one
    # boolean is not enough: five images and one image cost differently but would
    # look the same in the log.
    n_images: int = 0
    # How many prompt tokens the server took from the prefix cache
    # (`usage.prompt_tokens_details.cached_tokens`). vLLM returns the field only with
    # `--enable-prompt-tokens-details` and only when non-zero: `None` means "the
    # server did not say", covering both zero hits and a disabled flag.
    cached_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def truncated(self) -> bool:
        """The answer was cut off by the `max_tokens` cap, not finished by the model.

        Separate from `ok`: the call succeeded, the answer arrived and looks normal
        by every other sign; it is just its BEGINNING. Treating such an answer as
        misunderstood would mean curing the cap with the prompt.
        """
        return self.finish_reason == "length"

    @property
    def n_returned(self) -> int:
        return len(self.texts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "has_image": self.has_image,
            "n_images": self.n_images,
            "latency_sec": round(self.latency_sec, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "attempts": self.attempts,
            "n_requested": self.n_requested,
            "n_returned": self.n_returned,
            "error": self.error,
            "permanent": self.permanent,
            "context_overflow": self.context_overflow,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
        }


def pil_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def make_generation_kwargs(
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    seed: Optional[int],
) -> dict[str, Any]:
    """Request knobs for the generator. `temperature == 0` means greedy decoding.

    At zero, `top_p` and `top_k` are NOT sent, and this is not cosmetic: that is the
    request shape greedy measurements were taken with. Adding them here would change
    the measurement condition without changing the meaning.
    """
    if float(temperature) == 0.0:
        generation_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    else:
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_body": {"top_k": top_k},
        }

    if seed is not None:
        generation_kwargs["seed"] = seed

    return generation_kwargs


def call_vision(
    client: OpenAI,
    model_name: str,
    text: str,
    image: Image.Image | Sequence[Image.Image] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    n: int = 1,
    history: Sequence[dict[str, Any]] | None = None,
) -> LLMCall:
    """Call the endpoint, measuring cost, with a bounded number of retries.

    Returns a result rather than raising: a dropped connection is a normal run event,
    and the harness must see it as a failed step, not a crash.

    ``history`` is ready-made chat messages that go BEFORE the question (after the
    system prompt): a multi-turn conversation with `assistant` and `tool` roles.
    Images travel only in the last message, in `text` and `image`. Functions to call
    (`tools`, `tool_choice`) are passed in ``generation_kwargs`` like other request
    fields.

    ``n > 1`` asks for k answers in **one** request (`n` in the OpenAI-compatible
    API). Collecting k variants through k parallel requests would encode the image to
    base64, send it and have the model preprocess it k times instead of once, and
    the prefill would be recomputed for every request.
    """
    images = as_image_list(image)
    call = LLMCall(model=model_name, has_image=bool(images), n_images=len(images),
                   n_requested=max(1, int(n)))
    generation_kwargs = generation_kwargs or {}
    if call.n_requested > 1:
        generation_kwargs = {**generation_kwargs, "n": call.n_requested}
    started = time.monotonic()

    for attempt in range(1, max(1, max_attempts) + 1):
        call.attempts = attempt
        try:
            response = _request(client, model_name, text, image, generation_kwargs, system_prompt,
                                history)
            call.texts = [(choice.message.content or "").strip() for choice in response.choices]
            call.text = call.texts[0] if call.texts else ""
            # The field name depends on the vLLM version: `reasoning_content` in older
            # ones, `reasoning` in newer (`ChatMessage.reasoning`). Reading only the old
            # name lost the reasoning from the run log entirely: the field was empty
            # while the reasoning was not.
            call.reasonings = [
                (getattr(choice.message, "reasoning_content", None)
                 or getattr(choice.message, "reasoning", None) or "").strip()
                for choice in response.choices
            ]
            call.tool_calls_all = [_tool_calls_of(choice.message) for choice in response.choices]
            call.tool_calls = call.tool_calls_all[0] if call.tool_calls_all else []
            call.reasoning = call.reasonings[0] if call.reasonings else ""
            call.finish_reasons = [
                str(getattr(choice, "finish_reason", "") or "") for choice in response.choices
            ]
            call.finish_reason = call.finish_reasons[0] if call.finish_reasons else ""
            usage = getattr(response, "usage", None)
            if usage is not None:
                call.prompt_tokens = getattr(usage, "prompt_tokens", None)
                call.completion_tokens = getattr(usage, "completion_tokens", None)
                call.cached_tokens = _cached_tokens(usage)
            call.error = None
            break
        except Exception as exc:
            call.errors.append(repr(exc))
            call.error = repr(exc)
            call.permanent = is_permanent_error(exc)
            call.context_overflow = is_context_length_error(exc)
            if call.permanent:
                # Nothing to repeat: the server rejected the request itself, it did not lose
                # it. Retrying would cost three attempts and seconds of sleep at every
                # step of every part, all of it wall time.
                logger.warning(
                    "Request to %s rejected by the server, a retry will not help: %s", model_name, exc
                )
                break
            if attempt < max_attempts:
                logger.warning("Request to %s failed (attempt %d): %s", model_name, attempt, exc)
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    call.latency_sec = time.monotonic() - started
    return call


def _tool_calls_of(message: Any) -> list[dict[str, str]]:
    """Function calls of one answer as plain dicts, without client types."""
    calls = []
    for item in getattr(message, "tool_calls", None) or []:
        function = getattr(item, "function", None)
        if function is None:
            continue
        calls.append({
            "id": str(getattr(item, "id", "") or ""),
            "name": str(getattr(function, "name", "") or ""),
            "arguments": str(getattr(function, "arguments", "") or ""),
        })
    return calls


def as_image_list(image: Any) -> list[Image.Image]:
    """One image, a list of images or nothing, always as a list.

    Callers speak about images in two ways (one, and a list), while everything below
    (network, counters, length estimate) must count them the same way. The parsing of
    this pair lives in one place: two copies would diverge at the first caller that
    passes something unexpected.
    """
    if image is None:
        return []
    if isinstance(image, (list, tuple)):
        return [item for item in image if item is not None]
    return [image]


def _request(
    client: OpenAI,
    model_name: str,
    text: str,
    image: Any,
    generation_kwargs: dict[str, Any],
    system_prompt: str | None,
    history: Sequence[dict[str, Any]] | None = None,
) -> Any:
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or ())
    messages.append({"role": "user", "content": _encoded(_user_parts(text, as_image_list(image)))})
    return client.chat.completions.create(model=model_name, messages=messages, **generation_kwargs)


def _user_parts(text: str, images: list[Any], keep_last: bool = False) -> str | list[Any]:
    """User message content: a string, or parts (text and images as is).

    Images are not encoded yet: `prewarm_content` compares the parts of two requests
    by image identity, not by base64. `keep_last` keeps the last text piece even if
    it is only whitespace (needed by the prewarm).
    """
    pieces = text.split(IMAGE_SLOT)
    if len(pieces) > 1 and len(pieces) - 1 != len(images):
        # Markers do not match the images: use the default order (see `IMAGE_SLOT`).
        text = text.replace(IMAGE_SLOT, "")
        pieces = [text]
    if not images:
        return text
    if len(pieces) > 1:
        content: list[Any] = []
        for index, piece in enumerate(pieces):
            if piece.strip() or (keep_last and piece and index == len(pieces) - 1):
                content.append(piece)
            if index < len(images):
                content.append(images[index])
        return content
    # Images go BEFORE the text, in the order the caller gave them: the text refers
    # to them by number ("image 2 = c13"), so the order is part of the meaning of the
    # question, not a serialization detail. The server must run with
    # `limit-mm-per-prompt` no smaller than their count, or the request is rejected
    # whole.
    return [*images, text]


def _encoded(parts: str | list[Any]) -> str | list[dict[str, Any]]:
    if isinstance(parts, str):
        return parts
    return [{"type": "text", "text": part} if isinstance(part, str)
            else {"type": "image_url", "image_url": {"url": pil_image_to_data_url(part)}}
            for part in parts]


def _cached_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(details, dict):
        value = details.get("cached_tokens")
    else:
        value = getattr(details, "cached_tokens", None)
    return int(value) if value is not None else None


def prewarm_content(text: str, image: Any, prefix: str) -> str | list[dict[str, Any]] | None:
    """Content of a prewarm request: the beginning of the question `text` up to the end of `prefix`.

    **Why.** A hybrid model's prefix cache in `align` mode stores the GDN state only at
    the end of a prefill chunk: for the next turn to hit the cache on the unchanged
    part, a request that ends exactly there is needed. It is sent with
    `continue_final_message`: the chat template does not close the user message, and
    the rendered prewarm prompt must be **token for token** the beginning of the
    question prompt.

    Two obligations follow. The caller's: `prefix` ends at a boundary the tokenizer
    will not merge with the continuation (for the dialogue, `"\\n\\n"` between the
    transcript parts: without it, `.` at the end of a line and `.\\n\\n` in the
    question are different tokens). Ours: the prewarm parts match the question parts
    except the last one, which is the beginning of the corresponding question part.
    `continue_final_message` in `transformers` cuts the rendering after the LAST text
    block, so images at the very tail of the prefix are dropped here; otherwise the
    template would cut silently.

    `None` means there is no prefix: `text` does not start with it, the markers do
    not match the images (the question goes in the default order, images first), or
    the prefix has no text. No prewarm is made then.
    """
    if not prefix or not text.startswith(prefix):
        return None
    images = as_image_list(image)
    main = _user_parts(text, images)
    head = images[:prefix.count(IMAGE_SLOT)]
    for keep_last in (True, False):
        parts = _user_parts(prefix, head, keep_last=keep_last)
        if isinstance(parts, str) or isinstance(main, str):
            if isinstance(parts, str) and isinstance(main, str) and parts:
                return parts
            return None
        while parts and not isinstance(parts[-1], str):
            parts.pop()
        if not parts or len(parts) > len(main):
            continue
        last = len(parts) - 1
        same = all(a is b if not isinstance(a, str) else a == b
                   for a, b in zip(parts[:last], main[:last]))
        if same and isinstance(main[last], str) and main[last].startswith(parts[last]):
            return _encoded(parts)
    return None


def call_prewarm(
    client: OpenAI,
    model_name: str,
    content: str | list[dict[str, Any]],
    generation_kwargs: dict[str, Any],
    history: Sequence[dict[str, Any]] | None = None,
) -> LLMCall:
    """Prefix-cache prewarm: `content` as an open message, one answer token.

    `generation_kwargs` are the same as the question's (functions,
    `chat_template_kwargs`, reply header): the template renders the system block from
    them, and any difference shifts everything after it. One attempt, no retries: a
    failed prewarm costs only a cache hit, so waiting for retries is pointless.
    """
    extra = dict(generation_kwargs.get("extra_body") or {})
    extra.update(add_generation_prompt=False, continue_final_message=True)
    kwargs = {**generation_kwargs, "max_tokens": 1, "extra_body": extra}
    kwargs.pop("n", None)
    call = LLMCall(model=model_name)
    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[*(history or ()), {"role": "user", "content": content}],
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            call.prompt_tokens = getattr(usage, "prompt_tokens", None)
            call.completion_tokens = getattr(usage, "completion_tokens", None)
            call.cached_tokens = _cached_tokens(usage)
    except Exception as exc:
        call.error = repr(exc)
        call.errors.append(call.error)
    call.latency_sec = time.monotonic() - started
    return call


def send_vision_chat_request(
    client: OpenAI,
    model_name: str,
    image: Image.Image,
    text: str,
    generation_kwargs: dict[str, Any],
    system_prompt: str | None = None,
) -> str:
    """Legacy signature: returns only the answer text. Kept for simple call sites."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": pil_image_to_data_url(image)},
                },
                {"type": "text", "text": text},
            ],
        }
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **generation_kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def send_one_generate_request(
    client: OpenAI,
    model_name: str,
    item: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> str:
    return send_vision_chat_request(
        client=client,
        model_name=model_name,
        image=item["image"],
        text=item["text"],
        generation_kwargs=generation_kwargs,
    )

