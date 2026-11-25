# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

r"""Locate rows: the joint-v4 prompt that asks for verbatim spans.

A Locate row keeps the protocol v4 context: system line, ``"State:\\n"``
and the state, then the separator. It replaces the option-line chunks with
one chunk per Locate question, the checkpoint's own prompt line from
``locate-span/locate_config.json`` ending at the ``<|locate|>`` marker::

    [context] <|fim_middle|> [Question:\\nLocate verbatim spans:\\n
    <instructions> <|locate|>] [second question ...]

The row answers through the second readout channel: the span head scores
the state-token rows (its ``document``) against each marker row (its
``labels``), so the encoder names both sets of positions: everything
between the row's start and the separator belongs to ``document``. It
also keeps a character offset for every state-token row, because the
head answers in token indices and the response answers in text.

Escape rules and the shared-row segment machinery are the decisions
format's: the chunks hide from each other the same way, and the state
attends all of them, so one answer can mention a span another question
named.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from max.pipelines.architectures.qwen3_5_decider.span_head import LocateSettings
from max.pipelines.modeling.types.tokenizer import LabelScoringTokenizer

from ._decisions_prompt import DecisionText
from ._decisions_scoring import DecisionRequestError
from ._joint_v4_format import V4Renderer


class OffsetTokenizer(Protocol):
    """The extra tokenizer capability a Locate request needs.

    :class:`~max.pipelines.modeling.types.tokenizer.LabelScoringTokenizer`
    answers ids; a Locate answer is text, so the encoder also needs each
    token's character span. Implementations return the offset mapping of
    one encode call: ``ids`` and one ``(start, end)`` pair per id, into
    the same text that produced them.
    """

    async def encode_with_offsets(
        self, prompt: str, add_special_tokens: bool = True
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Encodes text to ids and their character spans."""
        ...


@dataclass(frozen=True)
class EncodedLocate:
    """One Locate row and everything the head and the route need about it.

    Row-local token indices throughout; the row is a single sequence, and
    the serving step scores one request per pass.
    """

    prompt_ids: list[int]
    """The whole row, ending at the last question's ``<|locate|>``."""

    state_positions: list[int]
    """The token indices the span head reads as its ``document``."""

    marker_positions: list[int]
    """The index of each question's ``<|locate|>``, in question order."""

    state_char_offsets: list[tuple[int, int]]
    """One ``(start, end)`` per :attr:`state_positions` entry, into the
    request's state text: the head's token spans become these."""

    segments: list[int]
    """The span lengths the segment machinery rewinds at: the context and
    separator first, then one entry per question chunk. The row's segment
    table has one boundary per entry; the state span attends every
    question, and each question sees the state and the questions before
    the layout's own rules allow, exactly as the decisions shared row."""

    def label_positions(self) -> list[int]:
        """The rows the ``label_hidden`` channel gathers, in head order.

        The ``document`` rows first, then the ``labels`` rows. A route
        splits the returned block at :data:`state_count`.
        """
        return self.state_positions + self.marker_positions

    @property
    def state_count(self) -> int:
        return len(self.state_positions)


async def encode_locate(
    renderer: V4Renderer,
    settings: LocateSettings,
    tokenizer: LabelScoringTokenizer,
    state: DecisionText,
    instructions: Sequence[str],
    separator_id: int,
) -> EncodedLocate:
    """Renders one Locate row: the context, the separator, one chunk each.

    Args:
        renderer: The decisions renderer, which owns the escape rules; a
            Locate row defuses special tokens exactly as a decisions row.
        settings: The checkpoint's Locate settings: prompt line, marker and
            marker id.
        tokenizer: Must answer :class:`OffsetTokenizer` as well; the answer
            is text, and text needs offsets.
        state: The state the questions point into.
        instructions: Each question's instruction text, in order.
        separator_id: The context separator the renderer put between the
            state and the question chunks, from the pinned check in
            :meth:`JointV4Format.separator_id`.

    Raises:
        DecisionRequestError: If the tokenizer cannot give offsets or the
            row does not fit the context length.
    """
    if not hasattr(tokenizer, "encode_with_offsets"):
        raise DecisionRequestError(
            "a locate question needs a tokenizer that answers "
            "encode_with_offsets; this deployment's does not"
        )
    encode_offsets: Any = tokenizer.encode_with_offsets

    context_text = renderer.context_text(
        state,
        lambda messages: tokenizer.apply_chat_template(
            messages, tools=None, add_generation_prompt=False
        ),
    )
    # The renderer promised the context ends with the escaped state, byte
    # for byte; its token rows are the ones the head reads. The row is
    # assembled as ``encode(prefix) + encode(state)``, and that split is
    # only legitimate when it agrees with one encode of the whole context:
    # a BPE merge across the state's first boundary would shift every
    # state token's row, so disagreement refuses the request rather than
    # answers with shifted rows. (The reference reads the same fact off an
    # offset mapping; ids are what this protocol exposes, and ids catch it.)
    escaped_state = renderer.escape(state)
    if not context_text.endswith(escaped_state):
        raise DecisionRequestError(
            "the chat template did not leave the state at the end of the "
            "context, so its token rows cannot be named"
        )
    prefix_text = context_text[: -len(escaped_state)]
    prefix_ids = await _encode_ids(tokenizer, prefix_text)
    state_ids, state_offsets = await encode_offsets(
        escaped_state, add_special_tokens=False
    )
    state_ids = [int(token) for token in state_ids]
    whole_ids = await _encode_ids(tokenizer, context_text)
    if prefix_ids + state_ids != whole_ids:
        raise DecisionRequestError(
            "the state does not tokenize the same inside the prompt and "
            "on its own, so its token rows would not line up; this "
            "tokenizer merged across the state boundary"
        )
    context_ids = prefix_ids + state_ids

    # The marker rides as its id, never as text: the release's tokenizer
    # registers no such added token (its added tokens stop below this id),
    # and the row's embedding is the adapter's trained marker vector, which
    # is why a Locate row needs the folded build. Encoding the literal would
    # split it into pieces and answer a different sequence than the one the
    # head was trained on.
    marker_id = settings.marker_id
    segments = [len(context_ids) + 1]
    row_ids = context_ids + [separator_id]
    marker_positions: list[int] = []
    for instruction in instructions:
        chunk_text = settings.prompt.format(
            instructions=renderer.escape(instruction)
        ).removesuffix(settings.marker)
        chunk_ids = await _encode_ids(tokenizer, chunk_text)
        chunk_ids.append(marker_id)
        segments.append(len(chunk_ids))
        row_ids.extend(chunk_ids)
        marker_positions.append(len(row_ids) - 1)

    max_length = tokenizer.max_length
    if max_length and len(row_ids) > max_length:
        raise DecisionRequestError(
            f"the locate prompt has {len(row_ids)} tokens, which does not "
            f"fit the context length of {max_length} tokens"
        )

    context_tokens = segments[0] - 1
    state_positions = list(
        range(
            context_tokens - len(state_ids),
            context_tokens,
        )
    )
    return EncodedLocate(
        prompt_ids=row_ids,
        state_positions=state_positions,
        marker_positions=marker_positions,
        state_char_offsets=[(int(a), int(b)) for a, b in state_offsets],
        segments=segments,
    )


async def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    ids = await tokenizer.encode(text, add_special_tokens=False)
    return [int(token) for token in ids]


def assemble_spans(
    candidates: Sequence[tuple[int, int, float]],
    state_char_offsets: Sequence[tuple[int, int]],
    state: str,
) -> list[tuple[int, int, str, float]]:
    """Token-index spans to text spans, the reference's postprocessing.

    ``candidates`` are ``(first_token, last_token, probability)`` pairs
    with inclusive token indices into the state rows. A span's characters
    run from the first token's start to the last token's end; leading and
    trailing whitespace is trimmed, and the same character pair from
    several candidates keeps its largest probability. Answers come back
    sorted by position.
    """
    unique: dict[tuple[int, int], float] = {}
    for first, last, probability in candidates:
        a = state_char_offsets[first][0]
        b = state_char_offsets[last][1]
        while a < b and state[a].isspace():
            a += 1
        while b > a and state[b - 1].isspace():
            b -= 1
        if a < b:
            unique[(a, b)] = max(probability, unique.get((a, b), 0.0))
    return [
        (start, end, state[start:end], probability)
        for (start, end), probability in sorted(unique.items())
    ]
