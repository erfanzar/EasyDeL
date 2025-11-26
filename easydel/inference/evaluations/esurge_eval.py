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

from eformer.loggings import get_logger

from easydel.infra.utils import ProcessingClassType

from ..esurge import eSurge

logger = get_logger("eSurgeLMEvalAdapter")

try:
    from lm_eval.api.model import LM  # type:ignore
except Exception as e:
    LM = object
    logger.warning(
        f"consider installing lm_eval if you want to use `eSurgeLMEvalAdapter` (err : {e}).",
        stacklevel=1,
    )


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
        self.model = None
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

        import easydel as ed

        if max_tokens is None:
            max_tokens = self.max_gen_toks

        if top_p is None:
            top_p = self.top_p

        if temperature is None:
            temperature = self.temperature

        # Create sampling params for each prompt
        sampling_params_list = []
        for i, _ in enumerate(prompts):
            current_stop_seq = None
            if stop_sequences and i < len(stop_sequences):
                current_stop_seq = stop_sequences[i]

            sampling_params_list.append(
                ed.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=current_stop_seq if current_stop_seq else [],
                    n=1,
                )
            )

        # Use eSurge's generate method
        results = self.surge.generate(
            prompts=prompts,
            sampling_params=sampling_params_list[0]
            if len(set(str(sp) for sp in sampling_params_list)) == 1
            else sampling_params_list,
            use_tqdm=True,
        )

        generated_texts = []
        for i, result in enumerate(results):
            text = result.get_text()
            # Apply stop sequences if present
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
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

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
        calculation might not be directly supported by the current `eSurge` setup.

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
        raise NotImplementedError("eSurgeLMEvalAdapter doesn't use greedy_until directly, use generate_until")
