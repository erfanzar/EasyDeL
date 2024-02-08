import requests
from typing import Optional, Literal


class EasyClient:
    def __init__(
            self,
            host: str = "0.0.0.0",
            port: Optional[int] = None,
            method: Literal["https", "http"] = "http"
    ):
        self.method = method
        self.host = host
        self.port = port

    def conversation_request(
            self,
            conversation: list[dict[str, str]],
            max_new_tokens: Optional[int] = None,
            greedy: bool = False,
            version: str = "v1"
    ):
        """
        The conversation_request function takes in a list of dictionaries, each containing the keys &quot;text&quot; and &quot;speaker&quot;,
        and returns a response object from the Hugging Face API. The response object contains information about the request,
        including whether it was successful or not (response.ok), as well as any error messages that may have been returned
        (response.json()). If successful, it also contains an array of generated responses to be used in your chatbot.

        :param self: Represent the instance of the class
        :param conversation: list[dict[str, str]]: Specify the conversation
        :param max_new_tokens: Optional[int]: Limit the number of new tokens that can be generated
        :param greedy: bool: Determine whether the model should generate a response that is generated is sampled
        (greedy=false) or one that is not sampled for generation (greedy=false)
        :param version: str: Specify which version of the serve_engine you want to use
        :return: A response object
        """
        port = ":" + str(self.port) if self.port is not None else ""
        response = requests.post(
            url=f"{self.method}://{self.host}{port}/generate/{version}/conversation",
            data={
                "conversation": conversation,
                "max_new_tokens": max_new_tokens,
                "greedy": greedy
            },
        )

        return response
