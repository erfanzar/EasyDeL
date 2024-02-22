from abc import ABC

from .base_prompter import BasePrompter
from typing import List, Optional


class GemmaPrompter(BasePrompter, ABC):
    def __init__(
            self,
    ):
        user_prefix = "<start_of_turn>user\n"
        assistant_prefix = "<start_of_turn>model\n"
        super().__init__(
            user_message_token=user_prefix,
            assistant_message_token=assistant_prefix,
            prompter_type="gemma",
            end_of_turn_token="<end_of_turn>\n",
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = ""
        for user, assistant in history:
            prompt += f"{self.user_message_token}{user}{self.end_of_turn_token}"
            prompt += f"{self.assistant_message_token}{assistant}{self.end_of_turn_token}"

        return prompt

    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: Optional[str]
    ) -> str:

        dialogs = prefix if prefix is not None else ""

        for user, assistant in history:
            dialogs += f"{self.user_message_token}{user}{self.end_of_turn_token}"
            dialogs += f"{self.assistant_message_token}{assistant}{self.end_of_turn_token}"

        dialogs += f"{self.user_message_token}{prompt}{self.end_of_turn_token}"
        dialogs += self.assistant_message_token
        return dialogs
