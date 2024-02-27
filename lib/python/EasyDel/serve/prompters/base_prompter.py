import abc
from typing import List, Any, Optional
from datetime import datetime
from abc import abstractmethod


class BasePrompter(abc.ABC):
    def __init__(
            self,
            prompter_type: str,
            user_message_token: str,
            assistant_message_token: str,
            end_of_turn_token: Optional[str] = None,
    ):
        self.prompter_type = prompter_type
        self.user_message_token = user_message_token
        self.assistant_message_token = assistant_message_token
        self.end_of_turn_token = end_of_turn_token

    @abstractmethod
    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ) -> str:
        raise NotImplementedError("NotImplementedYet !")

    @abstractmethod
    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: Optional[str]
    ) -> str:
        raise NotImplementedError("NotImplementedYet !")

    def content_finder(
            self,
            prompt: str,
            formatted_prompt: str,
            history: list[list[str]],
            system_message: str,
            external_data: str | Any
    ) -> str:
        raise NotImplementedError("NotImplementedYet !")

    def filter_response(
            self,
            response: str,
    ) -> str:
        response = response.replace(
            self.user_message_token, ""
        ).replace(
            self.assistant_message_token, ""
        )
        return response

    def get_stop_signs(self) -> List[str]:
        return [self.user_message_token, self.end_of_turn_token, self.assistant_message_token]

    def retrival_qa_template(
            self,
            question: str,
            contexts: list[str],
            base_question: Optional[str] = None,
            context_seperator_char: str = "\n"
    ):
        base_question = base_question or (
            "Use the following pieces of context to answer the question at the end. If you don't know the answer, "
            "just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}"
        )
        assert isinstance(contexts, list), "provide a list of strings"
        context = context_seperator_char.join(context for context in contexts)

        return self.user_message_token + base_question.format(
            context=context,
            question=question
        ) + self.assistant_message_token

    def __repr__(self):
        """
        The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        :param self: Refer to the instance of the class
        :return: A string representation of the object
        """
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                try:
                    repr_src = f"\t{k} : " + \
                               v.__str__().replace("\n", "\n\t") + "\n"
                    string += repr_src if len(
                        repr_src) < 500 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
                except TypeError:
                    ...
        return string + ")"

    def __str__(self):
        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
