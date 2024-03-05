from abc import ABC

from .base_prompter import BasePrompter
from typing import List, Optional


class ZephyrPrompter(BasePrompter, ABC):
    def __init__(
            self,
            end_of_turn_token="<|endoftext|>\n"
    ):

        user_prefix = "<|user|>\n"
        assistant_prefix = "<|assistant|>\n"
        super().__init__(
            user_message_token=user_prefix,
            assistant_message_token=assistant_prefix,
            prompter_type="zephyr",
            end_of_turn_token=end_of_turn_token,
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = ""
        for user, assistant in history:
            prompt += f"{self.user_message_token}{user} "
            prompt += f"{self.assistant_message_token}{assistant} "

        return prompt

    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: Optional[str]
    ) -> str:

        sys_pr = "" if system_message is None else f"<|system|>\n{system_message}{self.end_of_turn_token}"
        dialogs = prefix if prefix is not None else ""
        dialogs = sys_pr + dialogs

        for user, assistant in history:
            dialogs += f"{self.user_message_token}{user}"
            dialogs += f"{self.assistant_message_token}{assistant}"

        dialogs += f"{self.user_message_token}{prompt}"
        dialogs += self.assistant_message_token
        return dialogs
