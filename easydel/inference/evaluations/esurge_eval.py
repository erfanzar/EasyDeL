# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
from eformer.loggings import get_logger
from jax.scipy.special import logsumexp

from easydel.infra.utils import ProcessingClassType

from ..esurge import eSurge

logger = get_logger("eSurgeLMEvalAdapter")

try:
    from lm_eval.api.model import LM  # type:ignore
except Exception:
    LM = object
    err = "consider installing easydel[lm_eval] if you want to use `eSurgeLMEvalAdapter`."
    logger.warning(err, stacklevel=1)

_DEFAULT_REQUEST_ID_PREFIX = "lm_eval"


def _chunked(seq: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _trim_stop_sequences(text: str, stop: list[str]) -> str:
    if not stop:
        return text
    earliest = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx == -1:
            continue
        if earliest is None or idx < earliest:
            earliest = idx
    return text if earliest is None else text[:earliest]


def _get_rolling_token_windows(
    token_list: list[int],
    *,
    prefix_token: int,
    max_seq_len: int,
    context_len: int,
) -> Iterable[tuple[list[int], list[int]]]:
    """Port of `lm_eval.utils.get_rolling_token_windows` (v0.4.9.1)."""
    if not token_list:
        return
    if not (1 <= context_len <= max_seq_len):
        raise ValueError(f"context_len must be in [1, {max_seq_len}], got {context_len}")

    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens.
    first_seq_len = min(max_seq_len, len(token_list))
    yield [prefix_token, *token_list[: first_seq_len - 1]], token_list[:first_seq_len]
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len
        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def _make_disjoint_window(context: list[int], continuation: list[int]) -> tuple[list[int], list[int]]:
    """Port of `lm_eval.utils.make_disjoint_window` (v0.4.9.1)."""
    if not continuation:
        return context, continuation
    return context[: len(context) - (len(continuation) - 1)], continuation


class eSurgeLMEvalAdapter(LM):
    """Adapter for EasyDeL models to be compatible with lm-evaluation-harness.

    This class inherits from lm_eval.api.model.LM to ensure compatibility with the harness,
    allowing EasyDeL models to be evaluated using the lm-evaluation-harness framework.
    It wraps an `eSurge` instance for efficient inference with advanced features like
    smart bytecode decoding and context management.
    """

    def __init__(
        self,
        surge: eSurge,
        processor: ProcessingClassType,
        max_length: int = 8192,
        max_new_tokens: int = 2048,
        top_p: float = 0.95,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ):
        """Initializes the eSurgeLMEvalAdapter.

        Args:
            surge: An instance of `eSurge` for model inference.
            processor: The tokenizer/processor associated with the model.
            max_length: The maximum context length for the model. Defaults to 8192.
            max_new_tokens: Maximum number of tokens to generate. Defaults to 2048.
            top_p: Top-p sampling parameter. Defaults to 0.95.
            temperature: Sampling temperature. Defaults to 0.0 (greedy).
            batch_size: Optional batch size override. If None, uses surge's max_num_seqs.
        """
        super().__init__()
        self.max_length = max_length
        self.tokenizer = processor
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        self.surge = surge
        self._batch_size = batch_size or surge.max_num_seqs
        self.model = getattr(getattr(surge, "runner", None), "model", None)
        self.setup_complete = False
        self._setup()

    def _setup(self):
        """Set up the eSurge engine.

        Ensures the eSurge scheduler is running; it should auto-init at construction,
        but we guard against any cases where it is not.
        """
        if self.setup_complete:
            return

        # eSurge automatically starts scheduler in __init__
        # Just verify it's running
        if not self.surge._scheduler_running:
            self.surge.initiate()

        self.setup_complete = True

    def stop(self):
        """Stop the eSurge engine.

        Terminates the underlying `eSurge` scheduler thread.
        """
        if self.surge:
            self.surge.terminate()

    def _generate(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[list[str]] | None = None,
    ) -> list[str]:
        """Generate responses for a list of prompts.

        Args:
            prompts: List of prompts to generate responses for.
            max_tokens: Maximum number of tokens to generate per prompt.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            stop_sequences: List of lists of stop sequences, one list per prompt.
                            Generation stops if any sequence in the corresponding list is encountered.

        Returns:
            List of generated responses.
        """
        if not self.setup_complete:
            self._setup()

        from easydel.inference.sampling_params import SamplingParams

        if max_tokens is None:
            max_tokens = self.max_gen_toks

        if top_p is None:
            top_p = self.top_p

        if temperature is None:
            temperature = self.temperature

        if not prompts:
            return []

        normalized_stops: list[list[str]] = []
        if stop_sequences is None:
            normalized_stops = [[] for _ in prompts]
        else:
            for i in range(len(prompts)):
                stops = stop_sequences[i] if i < len(stop_sequences) else []
                normalized_stops.append([str(s) for s in (stops or [])])

        # eSurge doesn't accept per-prompt SamplingParams; group by stop sequences.
        groups: dict[tuple[int, float, float, tuple[str, ...]], list[int]] = {}
        for i, stops in enumerate(normalized_stops):
            key = (int(max_tokens), float(temperature), float(top_p), tuple(stops))
            groups.setdefault(key, []).append(i)

        outputs: list[str] = [""] * len(prompts)
        for (mt, temp, tp, stops_tuple), indices in groups.items():
            group_prompts = [prompts[i] for i in indices]
            request_ids = [f"{_DEFAULT_REQUEST_ID_PREFIX}-{uuid.uuid4().hex}" for _ in group_prompts]
            sampling_params = SamplingParams(
                max_tokens=mt,
                temperature=temp,
                top_p=tp,
                stop=list(stops_tuple),
                n=1,
            )

            # eSurge.generate may return outputs in completion order; re-map via request_id.
            results = self.surge.generate(
                prompts=group_prompts,
                sampling_params=sampling_params,
                request_id=request_ids,
                use_tqdm=False,
            )
            by_id = {r.request_id: r for r in results}
            for prompt_index, request_id in zip(indices, request_ids, strict=False):
                if request_id not in by_id:
                    raise RuntimeError(
                        f"eSurge.generate returned {len(results)}/{len(request_ids)} outputs; missing {request_id!r}."
                    )
                text = by_id[request_id].get_text()
                outputs[prompt_index] = _trim_stop_sequences(text, list(stops_tuple))

        return outputs

    def _extract_choice_from_generation(self, generation: str) -> str:
        """Extract a multiple-choice answer (A, B, C, D) from generated text.

        Args:
            generation: The generated text string.

        Returns:
            The extracted choice (e.g., "A", "B", "C", "D") or an empty string if no clear choice is found.
        """
        import re

        patterns = [
            r"^([A-Da-d])[^A-Za-z0-9]",
            r"^([A-Da-d])$",
            r"[Aa]nswer[^A-Za-z0-9]*([A-Da-d])",
            r"[Tt]he answer is[^A-Za-z0-9]*([A-Da-d])",
            r"[Oo]ption[^A-Za-z0-9]*([A-Da-d])",
            r"[Cc]hoice[^A-Za-z0-9]*([A-Da-d])",
        ]

        for pattern in patterns:
            match = re.search(pattern, generation.strip())
            if match:
                return match.group(1).upper()

        first_char = generation.strip()[0:1].upper()
        if first_char in "ABCD":
            return first_char

        return ""

    def generate_until(self, instances):
        """
        Generate text until a specified set of stop sequences is reached for each instance.

        This method is part of the lm-evaluation-harness LM interface.

        Args:
            instances: List of Instance objects from lm-evaluation-harness.
                       Each instance is expected to contain the prompt as the first argument
                       and an optional dictionary as the second argument with a 'until' key
                       containing a list of stop sequences.

        Returns:
            List of generated strings, one for each instance.
        """
        prompts: list[str] = []
        stop_sequences: list[list[str]] = []

        for instance in instances:
            if hasattr(instance, "arguments"):
                arguments = instance.arguments
            else:
                arguments = instance

            prompt = str(arguments[0]) if arguments else ""
            config = arguments[1] if len(arguments) > 1 else None
            if isinstance(config, dict):
                until = config.get("until", []) or []
            elif isinstance(config, list):
                until = config
            else:
                until = []

            prompts.append(prompt)
            stop_sequences.append([str(s) for s in until])

        return self._generate(
            prompts,
            max_tokens=self.max_gen_toks,
            stop_sequences=stop_sequences,
        )

    @property
    def eot_token_id(self):
        """Get the end-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """Get the maximum context length."""
        return self._max_length

    @max_length.setter
    def max_length(self, value):
        """Set the maximum context length."""
        self._max_length = value

    @property
    def max_gen_toks(self):
        """Get the maximum number of tokens to generate."""
        return self.max_new_tokens

    @property
    def batch_size(self):
        """Get the batch size."""
        return self._batch_size

    @property
    def device(self):
        """Get the device (CPU/GPU)."""
        return "cpu"

    @property
    def tokenizer_name(self):
        """Get the tokenizer name for chat template support.

        Returns the name or path of the tokenizer/model being used.
        This is required by lm_eval for proper chat template handling.
        """
        # Try to get the tokenizer name from various possible attributes
        if hasattr(self.tokenizer, "name_or_path") and self.tokenizer.name_or_path:
            return self.tokenizer.name_or_path
        elif hasattr(self.tokenizer, "tokenizer_name") and self.tokenizer.tokenizer_name:
            return self.tokenizer.tokenizer_name
        elif hasattr(self.tokenizer, "__class__"):
            # Return the class name as a fallback
            return self.tokenizer.__class__.__name__
        else:
            return ""

    def apply_chat_template(self, messages, add_generation_prompt: bool):
        """Apply chat template to messages.

        This method is required by lm_eval for chat-based evaluations.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            String with the formatted chat template applied
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

        lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in (messages or [])]
        if add_generation_prompt:
            lines.append("assistant:")
        return "\n".join(lines)

    def tok_encode(self, string: str):
        """Encode a string into token IDs.

        Args:
            string: The input string.

        Returns:
            A list of token IDs.
        """
        try:
            return self.tokenizer.encode(string, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        """Decode token IDs into a string.

        Args:
            tokens: A list or tensor of token IDs.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)

    def _encode_text(self, text: str) -> list[int]:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            ids = encoded.get("input_ids", [])
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(t) for t in ids]
        except Exception:
            return [int(t) for t in self.tok_encode(text)]

    def _loglikelihood_token_ids(
        self,
        context_token_ids: list[list[int]],
        continuation_token_ids: list[list[int]],
    ) -> list[tuple[float, bool]]:
        if self.model is None:
            raise RuntimeError("Scoring model is not available on the eSurge engine (missing `surge.runner.model`).")

        if len(context_token_ids) != len(continuation_token_ids):
            raise ValueError("context_token_ids and continuation_token_ids must have the same length")

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(pad_id, list):
            pad_id = pad_id[0] if pad_id else 0
        if pad_id is None:
            pad_id = 0
        pad_id = int(pad_id)

        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if isinstance(bos_id, list):
            bos_id = bos_id[0] if bos_id else None
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, list):
            eos_id = eos_id[0] if eos_id else None

        # Prepare (possibly truncated) token sequences (mirror lm-eval's causal logic).
        # For causal scoring, we keep the full continuation (<= max_length) and truncate the context from the left so
        # that `len(context)+len(continuation) <= max_length+1`. The model is run on `tokens[:-1]` (<= max_length).
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        context_lens: list[int] = []
        cont_lens: list[int] = []

        for ctx, cont in zip(context_token_ids, continuation_token_ids, strict=False):
            ctx_ids = [int(t) for t in (ctx or [])]
            cont_ids = [int(t) for t in (cont or [])]

            if not ctx_ids:
                if bos_id is not None:
                    ctx_ids = [int(bos_id)]
                elif eos_id is not None:
                    ctx_ids = [int(eos_id)]
                else:
                    ctx_ids = [pad_id]

            if len(cont_ids) > self.max_length:
                raise ValueError(
                    f"Continuation is too long for scoring (len={len(cont_ids)}) with max_length={self.max_length}."
                )

            total_len = len(ctx_ids) + len(cont_ids)
            if total_len > self.max_length + 1:
                # Keep the full continuation and truncate context from the left.
                drop = total_len - (self.max_length + 1)
                if drop >= len(ctx_ids):
                    raise ValueError(
                        "Context truncation would drop the entire context; "
                        f"got ctx={len(ctx_ids)}, cont={len(cont_ids)}, max_length={self.max_length}."
                    )
                ctx_ids = ctx_ids[drop:]

            full = ctx_ids + cont_ids
            if len(full) < 2:
                # No next-token targets; score is trivially 0.
                inputs.append([])
                targets.append([])
                context_lens.append(len(ctx_ids))
                cont_lens.append(len(cont_ids))
                continue

            context_lens.append(len(ctx_ids))
            cont_lens.append(len(cont_ids))

            # Shift for next-token prediction.
            inputs.append(full[:-1])
            targets.append(full[1:])

        results: list[tuple[float, bool]] = [(0.0, True) for _ in inputs]
        scored_indices = [i for i, n in enumerate(cont_lens) if n > 0]
        if not scored_indices:
            return results

        inputs_scored = [inputs[i] for i in scored_indices]
        targets_scored = [targets[i] for i in scored_indices]
        context_lens_scored = [context_lens[i] for i in scored_indices]
        cont_lens_scored = [cont_lens[i] for i in scored_indices]

        max_len = max((len(x) for x in inputs_scored), default=0)
        if max_len < 1:
            return results

        import numpy as np

        input_ids = np.full((len(inputs_scored), max_len), pad_id, dtype=np.int32)
        target_ids = np.full((len(inputs_scored), max_len), pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(inputs_scored), max_len), dtype=bool)

        for i, ids in enumerate(inputs_scored):
            input_ids[i, : len(ids)] = np.asarray(ids, dtype=np.int32)
            target_ids[i, : len(ids)] = np.asarray(targets_scored[i], dtype=np.int32)
            attention_mask[i, : len(ids)] = True

        input_ids_jax = jnp.asarray(input_ids)
        target_ids_jax = jnp.asarray(target_ids)
        attention_mask_jax = jnp.asarray(attention_mask)

        outputs = self.model(input_ids=input_ids_jax, attention_mask=attention_mask_jax)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model forward did not return logits; cannot compute loglikelihood.")

        log_denom = logsumexp(logits, axis=-1)
        selected = jnp.take_along_axis(logits, target_ids_jax[..., None], axis=-1).squeeze(-1)
        token_logprobs = selected - log_denom

        greedy_ids = jnp.argmax(logits, axis=-1)

        positions = jnp.arange(max_len)[None, :]
        start = (jnp.asarray(context_lens_scored) - 1)[:, None]
        end = (jnp.asarray(context_lens_scored) - 1 + jnp.asarray(cont_lens_scored))[:, None]
        score_mask = (positions >= start) & (positions < end) & attention_mask_jax

        seq_loglikelihood = jnp.sum(jnp.where(score_mask, token_logprobs, 0.0), axis=-1)
        seq_is_greedy = jnp.all(jnp.where(score_mask, greedy_ids == target_ids_jax, True), axis=-1)

        for out_i, (ll, is_g) in enumerate(zip(seq_loglikelihood, seq_is_greedy, strict=False)):
            results[scored_indices[out_i]] = (float(ll), bool(is_g))
        return results

    def _model_call(self, inps):
        """
        This method is not directly used by eSurgeLMEvalAdapter but is required by the LM interface.

        In our case, loglikelihood and greedy_until handle the model calls directly
        by interacting with the `eSurge` instance.

        Raises:
            NotImplementedError: This method is not implemented as it's not used.
        """
        raise NotImplementedError("eSurgeLMEvalAdapter doesn't use _model_call directly")

    def _model_generate(self, context, max_length, eos_token_id):
        """Generate text from context.

        This method is not directly used by eSurgeLMEvalAdapter but is required by the LM interface.
        Generation is handled by the `_generate` and `generate_until` methods.

        Args:
            context: The input context.
            max_length: The maximum length of the generated sequence.
            eos_token_id: The end-of-sequence token ID.

        Raises:
            NotImplementedError: This method is not implemented as it's not used.
        """
        raise NotImplementedError("eSurgeLMEvalAdapter doesn't use _model_generate directly")

    def loglikelihood(self, instances):
        """
        Compute log-likelihood of completions given contexts.

        This method is part of the lm-evaluation-harness LM interface.
        It computes the exact teacher-forced log-likelihood of the continuation tokens.

        Args:
            instances: List of Instance objects from lm-evaluation-harness.
                       For multiple-choice tasks, instances are expected to have context and continuation.

        Returns:
            List of (log_likelihood, is_greedy) tuples.
        """
        requests: list[tuple[str, str]] = []

        for instance in instances:
            if hasattr(instance, "arguments"):
                arguments = instance.arguments
            else:
                arguments = instance

            if len(arguments) >= 2:
                requests.append((str(arguments[0]), str(arguments[1])))
            else:
                logger.warning("Invalid loglikelihood instance format: %s", instance)
                requests.append(("", ""))

        results: list[tuple[float, bool]] = []
        for chunk in _chunked(requests, self._batch_size):
            ctx_ids = [self._encode_text(ctx) for ctx, _ in chunk]
            cont_ids = [self._encode_text(cont) for _, cont in chunk]
            results.extend(self._loglikelihood_token_ids(ctx_ids, cont_ids))

        return results

    def loglikelihood_rolling(self, instances):
        """
        Calculate log-likelihood of token sequences in a rolling fashion.

        This method is part of the lm-evaluation-harness LM interface.

        Args:
            instances: List of Instance objects from lm-evaluation-harness.
                       Instances contain either the raw string or a pre-tokenized sequence.

        Returns:
            List of rolling log-likelihoods, one per instance.
        """
        per_request_windows: list[tuple[int, list[int], list[int]]] = []

        prefix = self.eot_token_id
        max_seq_len = int(self.max_length)
        context_len = 1  # matches lm-eval default for rolling perplexity

        for req_idx, instance in enumerate(instances):
            if hasattr(instance, "arguments"):
                arguments = instance.arguments
            else:
                arguments = instance

            if not arguments:
                continue

            seq = arguments[0]
            token_ids = [int(t) for t in seq] if isinstance(seq, list) else self.tok_encode(str(seq))
            for ctx, cont in _get_rolling_token_windows(
                token_ids,
                prefix_token=int(prefix),
                max_seq_len=max_seq_len,
                context_len=context_len,
            ):
                ctx_disjoint, cont_disjoint = _make_disjoint_window(ctx, cont)
                per_request_windows.append((req_idx, ctx_disjoint, cont_disjoint))

        if not instances:
            return []

        totals = [0.0 for _ in range(len(instances))]
        if not per_request_windows:
            return totals

        window_ctx = [w[1] for w in per_request_windows]
        window_cont = [w[2] for w in per_request_windows]

        for start in range(0, len(per_request_windows), self._batch_size):
            batch_ctx = window_ctx[start : start + self._batch_size]
            batch_cont = window_cont[start : start + self._batch_size]
            batch_scores = self._loglikelihood_token_ids(batch_ctx, batch_cont)

            for (req_idx, _ctx, _cont), (ll, _is_greedy) in zip(
                per_request_windows[start : start + self._batch_size],
                batch_scores,
                strict=False,
            ):
                totals[req_idx] += float(ll)

        return totals

    def greedy_until(self, requests):
        """
        Generate completions for prompts until a stop sequence is reached using greedy decoding.

        This method is part of the lm-evaluation-harness LM interface.
        It currently raises NotImplementedError as its functionality is covered by `generate_until`.

        Args:
            requests: List of (context, stopping_sequences) tuples.

        Returns:
            List of generated completions.

        """
        # Support both legacy (context, until) tuples and modern Instance objects.
        if requests and not hasattr(requests[0], "arguments") and isinstance(requests[0], tuple):
            prompts = [str(r[0]) for r in requests]
            stop_sequences = [[str(s) for s in (r[1] or [])] for r in requests]
        else:
            prompts = []
            stop_sequences = []
            for instance in requests:
                if hasattr(instance, "arguments"):
                    arguments = instance.arguments
                else:
                    arguments = instance
                prompt = str(arguments[0]) if arguments else ""
                config = arguments[1] if len(arguments) > 1 else None
                if isinstance(config, dict):
                    until = config.get("until", []) or []
                elif isinstance(config, list):
                    until = config
                else:
                    until = []
                prompts.append(prompt)
                stop_sequences.append([str(s) for s in until])

        return self._generate(
            prompts,
            max_tokens=self.max_gen_toks,
            temperature=0.0,
            top_p=1.0,
            stop_sequences=stop_sequences,
        )
