from abc import ABC
from typing import Optional
from .base_prompter import BasePrompter


class CargoPrompter(BasePrompter, ABC):
    def __init__(
            self,
            user_name: str = "USER",
            assistant_name: str = "GPT",
    ):
        user_prefix = f"\n{user_name} => "
        assistant_prefix = f"\n{assistant_name} => "
        super().__init__(
            user_message_token=user_prefix,
            assistant_message_token=assistant_prefix,
            prompter_type="cargo",
            end_of_turn_token="<end_of_turn>",
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = system_message + "\n\n"
        for user, assistant in history:
            prompt += f"{self.user_message_token}{user}"
            prompt += f"{self.assistant_message_token}{assistant}"

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
            dialogs += f"{self.user_message_token}{user}"
            dialogs += f"{self.assistant_message_token}{assistant}"

        dialogs += f"{self.user_message_token}{prompt}{self.end_of_turn_token}"
        dialogs += self.assistant_message_token
        return dialogs
