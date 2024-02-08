from abc import ABC
from typing import Optional

from .base_prompter import BasePrompter


class GuanacoPrompter(BasePrompter, ABC):
    def __init__(
            self,
    ):
        user_prefix = f"\n###Human: "
        assistant_prefix = f"\n###Assistant: "
        super().__init__(
            user_message_token=user_prefix,
            assistant_message_token=assistant_prefix,
            prompter_type="guanaco",
            end_of_turn_token="<|endoftext|>",
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = system_message + "\n\n"
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
