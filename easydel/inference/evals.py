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

import asyncio

from eformer.common_types import NOT_GIVEN, _Empty
from eformer.loggings import get_logger

from easydel.infra.utils import ProcessingClassType

from .vsurge import vSurge

logger = get_logger("vSurgeLMEvalAdapter")

try:
    from lm_eval.api.model import LM  # type:ignore
except Exception as e:
    LM = object
    logger.warning(
        f"consider installing lm_eval if you want to use `vSurgeLMEvalAdapter` (err : {e}).",
        stacklevel=1,
    )


class vSurgeLMEvalAdapter(LM):
    """Adapter for EasyDeL models to be compatible with lm-evaluation-harness.

    This class inherits from lm_eval.api.model.LM to ensure compatibility with the harness,
    allowing EasyDeL models to be evaluated using the lm-evaluation-harness framework.
    It wraps a `vSurge` instance for efficient inference.
    """

    def __init__(
        self,
        surge: vSurge,
        processor: ProcessingClassType,
        max_length: int = 8192,
        max_new_tokens: int = 2048,
        top_p: float = 0.95,
        temperature: float = 0.0,
    ):
        """Initializes the vSurgeLMEvalAdapter.

        Args:
            surge: An instance of `vSurge` for model inference.
            processor: The tokenizer/processor associated with the model.
            max_length: The maximum context length for the model. Defaults to 8192.
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
        self.model = None
        self.setup_complete = False
        self._setup()

    def _setup(self):
        """Set up the EasyDeL model synchronously.

        This method ensures the asynchronous setup (`_async_setup`) is run
        in a separate thread if the current event loop is already running,
        or directly if no loop is running.
        """
        if self.setup_complete:
            return

        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._async_setup())
                future.result()
        else:
            asyncio.run(self._async_setup())

    async def _async_setup(self):
        """Set up the EasyDeL model asynchronously.

        This method starts and compiles the `vSurge` engine.
        """
        if self.setup_complete:
            return

        self.surge.compile()
        self.surge.start()
        self.setup_complete = True

    def stop(self):
        """Stop the EasyDeL engine.

        Shuts down the underlying `vSurge` instance.
        """
        if self.surge:
            self.surge.stop()

    async def _generate_async(
        self,
        prompts: list[str],
        max_tokens: int | _Empty = NOT_GIVEN,
        temperature: float | _Empty = NOT_GIVEN,
        top_p: float | _Empty = NOT_GIVEN,
        stop_sequences: list[list[str]] | None = None,
    ) -> list[str]:
        """Generate responses for a list of prompts asynchronously.

        Args:
            prompts: List of prompts to generate responses for.
            max_tokens: Maximum number of tokens to generate per prompt. Defaults to 128.
            temperature: Sampling temperature. Defaults to 0.0 (greedy decoding).
            top_p: Top-p sampling parameter. Defaults to 1.0.
            stop_sequences: List of lists of stop sequences, one list per prompt.
                            Generation stops if any sequence in the corresponding list is encountered.
                            Defaults to None.

        Returns:
            List of generated responses, with the prompt text removed.
        """
        if not self.setup_complete:
            await self._async_setup()
        import easydel as ed

        if max_tokens is NOT_GIVEN:
            max_tokens = self.max_gen_toks

        if top_p is NOT_GIVEN:
            top_p = self.top_p

        if temperature is NOT_GIVEN:
            temperature = self.temperature

        sampling_params_list = []
        for i, _ in enumerate(prompts):
            current_stop_seq = None
            if stop_sequences and i < len(stop_sequences):
                current_stop_seq = stop_sequences[i]

            sampling_params_list.append(
                ed.SamplingParams(
                    max_tokens=max_tokens,  # This should probably be self.max_gen_toks or a passed arg
                    temperature=temperature,
                    top_p=top_p,
                    stop=current_stop_seq if current_stop_seq else [],  # Ensure it's a list
                    n=1,
                )
            )

        results = await self.surge.generate(
            prompts=prompts,
            sampling_params=sampling_params_list,
            stream=False,
        )
        generated_texts = []
        for i, result_list in enumerate(results):
            text = result_list.text[0][0]
            if stop_sequences and i < len(stop_sequences):
                for stop in stop_sequences[i]:
                    if stop in text:
                        text = text[: text.find(stop)]
            generated_texts.append(text)

        assert len(generated_texts) == len(prompts), (
            f"Mismatch between prompts sent ({len(prompts)}) and results received "
            f"({len(generated_texts)}) from surge.generate!."
        )

        return generated_texts

    def _generate(
        self,
        prompts: list[str],
        max_tokens: int | _Empty = NOT_GIVEN,
        temperature: float | _Empty = NOT_GIVEN,
        top_p: float | _Empty = NOT_GIVEN,
        stop_sequences: list[list[str]] | None = None,
    ) -> list[str]:
        """Generate responses for a list of prompts synchronously.

        This method is a synchronous wrapper around `_generate_async`.

        Args:
            prompts: List of prompts to generate responses for.
            max_tokens: Maximum number of tokens to generate per prompt. Defaults to 128.
            temperature: Sampling temperature. Defaults to NOT_GIVEN.
            top_p: Top-p sampling parameter. Defaults to NOT_GIVEN.
            stop_sequences: List of lists of stop sequences, one list per prompt.
                            Generation stops if any sequence in the corresponding list is encountered.
                            Defaults to None.

        Returns:
            List of generated responses, with the prompt text removed.
        """
        out = asyncio.run(
            self._generate_async(
                prompts,
                max_tokens,
                temperature,
                top_p,
                stop_sequences,
            )
        )
        return out

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

        requests = []
        for instance in instances:
            prompt = instance.arguments[0]

            if len(instance.arguments) > 1 and isinstance(instance.arguments[1], dict):
                config = instance.arguments[1]
                stop_sequences = config.get("until", [])
            else:
                stop_sequences = []

            requests.append((prompt, stop_sequences))

        generations = self._generate(
            [req[0] for req in requests],
            max_tokens=self.max_gen_toks,
            stop_sequences=[req[1] for req in requests],
        )

        return generations

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
        return self.max_concurrent_decodes

    @property
    def device(self):
        """Get the device (CPU/GPU)."""
        return "cpu"

    def tok_encode(self, string: str):
        """Encode a string into token IDs.

        Args:
            string: The input string.

        Returns:
            A list of token IDs.
        """
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        """Decode token IDs into a string.

        Args:
            tokens: A list or tensor of token IDs.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        This method is not directly used by vSurgeLMEvalAdapter but is required by the LM interface.

        In our case, loglikelihood and greedy_until handle the model calls directly
        by interacting with the `vSurge` instance.

        Raises:
            NotImplementedError: This method is not implemented as it's not used.
        """
        raise NotImplementedError("vSurgeLMEvalAdapter doesn't use _model_call directly")

    def _model_generate(self, context, max_length, eos_token_id):
        """Generate text from context.

        This method is not directly used by vSurgeLMEvalAdapter but is required by the LM interface.
        Generation is handled by the `_generate` and `generate_until` methods.

        Args:
            context: The input context.
            max_length: The maximum length of the generated sequence.
            eos_token_id: The end-of-sequence token ID.

        Raises:
            NotImplementedError: This method is not implemented as it's not used.
        """
        raise NotImplementedError("vSurgeLMEvalAdapter doesn't use _model_generate directly")

    def loglikelihood(self, instances):
        """
        Compute log-likelihood of completions given contexts.

        This method is part of the lm-evaluation-harness LM interface.
        It currently provides a placeholder implementation, especially for non-multiple-choice tasks.

        Args:
            instances: List of Instance objects from lm-evaluation-harness.
                       For multiple-choice tasks, instances are expected to have context and continuation.

        Returns:
            List of (log_likelihood, is_greedy) tuples.
                For multiple-choice tasks, log-likelihood is high if the extracted choice matches the continuation,
                low otherwise. For other tasks, a placeholder value is returned.
        """
        requests = []
        for instance in instances:
            if len(instance.arguments) >= 2:
                context = instance.arguments[0]
                continuation = instance.arguments[1]
                requests.append((context, continuation))
            else:
                print(f"Warning: Invalid instance format: {instance}")
                requests.append(("", ""))
        contexts = [req[0] for req in requests]
        continuations = [req[1] for req in requests]

        is_mc_task = False
        if contexts and len(contexts) > 0:
            mc_pattern = r"[A-D]\.\s"
            import re

            if any(re.search(mc_pattern, ctx) for ctx in contexts):
                is_mc_task = True

        results = []

        if is_mc_task:
            choices = "ABCD"
            max_tokens = 5
            generations = self._generate(contexts, max_tokens=max_tokens)

            for _i, (_, continuation, generation) in enumerate(zip(contexts, continuations, generations, strict=False)):
                predicted_choice = self._extract_choice_from_generation(generation)

                expected_choice = continuation.strip().upper()
                if expected_choice and expected_choice[0] in choices:
                    expected_choice = expected_choice[0]

                log_likelihood = 0.0 if predicted_choice == expected_choice else -100.0
                is_greedy = predicted_choice == expected_choice

                results.append((log_likelihood, is_greedy))
        else:
            for _ in zip(contexts, continuations, strict=False):
                results.append((-1.0, True))

        return results

    def loglikelihood_rolling(self, instances):
        """
        Calculate log-likelihood of token sequences in a rolling fashion.

        This method is part of the lm-evaluation-harness LM interface.
        It currently provides a placeholder implementation as actual rolling log-likelihood
        calculation might not be directly supported by the current `vSurge` setup.

        Args:
            instances: List of Instance objects from lm-evaluation-harness.
                       Instances are expected to contain the token sequence as the first argument.

        Returns:
            List of lists of (loglikelihood, is_greedy) pairs, one inner list per instance.
            Each inner list contains pairs for each token in the sequence (except the first).
            Currently returns placeholder values.
        """

        token_lists = []
        for instance in instances:
            if len(instance.arguments) >= 1 and isinstance(instance.arguments[0], list):
                tokens = instance.arguments[0]
                token_lists.append(tokens)
            else:
                print(f"Warning: Invalid instance format for rolling loglikelihood: {instance}")
                token_lists.append([])

        results = []

        for tokens in token_lists:
            token_results = []
            for _i in range(1, len(tokens)):
                log_likelihood = -2.0
                is_greedy = True
                token_results.append((log_likelihood, is_greedy))

            results.append(token_results)

        return results

    def greedy_until(self, requests):
        """
        Generate completions for prompts until a stop sequence is reached using greedy decoding.

        This method is part of the lm-evaluation-harness LM interface.
        It currently raises NotImplementedError as its functionality is covered by `generate_until`.

        Args:
            requests: List of (context, stopping_sequences) tuples.

        Returns:
            List of generated completions.

        Raises:
            NotImplementedError: This method is not implemented as `generate_until` provides similar functionality.
        """
        raise NotImplementedError("vSurgeLMEvalAdapter doesn't use greedy_until directly, use generate_until")
