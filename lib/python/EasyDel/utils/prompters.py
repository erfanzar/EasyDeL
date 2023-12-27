import typing


def antitoxin_prompter(
        history: typing.List[str],
        prompt: str,
        system: typing.Optional[str] = None,
):
    """
    The antitoxin_prompter function takes in a history of user-assistant interactions,
    a prompt from the user, and optionally a system response. It returns an input string
    that can be fed into the antitoxin model to generate an assistant response.

    :param history: typing.List[str]: Pass in the history of the conversation
    :param prompt: str: Pass the user's input to the assistant
    :param system: typing.Optional[str]: Pass the system's response to the prompt
    :param : Store the history of user and assistant interaction
    :return: A string that contains the user's prompt,
    
    """
    sys_str = f"<|im_start|>system\n{system}<|im_end|>\n" if system is not None else ""
    histories = ""
    for user, assistance in history:
        histories += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistance}<|im_end|>\n"
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return sys_str + histories + text


def antitoxin_prompter_chat_format(
        history: typing.List[str],
        system: typing.Optional[str] = None,
):
    """
    The antitoxin_prompter_chat_format function takes a list of strings and returns a string.
    The input is the history of the chat, which is a list of tuples where each tuple contains two strings:
    the user's message and the assistant's response. The output is formatted as follows:

    :param history: typing.List[str]: Pass in the history of user and assistant messages
    :param system: typing.Optional[str]: Pass in the system message
    :param : Store the history of the conversation
    :return: A string that contains the system message and
    
    """
    sys_str = f"<|im_start|>system\n{system}<|im_end|>\n" if system is not None else ""
    histories = ""
    for user, assistance in history:
        histories += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistance}<|im_end|>\n"
    return sys_str + histories


def llama2_prompter(
        history: typing.List[str],
        prompt: str,
        system: typing.Optional[str] = None,

):
    """
    The llama2_prompter function takes a history of user-system interactions,
    a prompt for the next system response, and optionally a system response.
    It returns an LLAMA2 formatted string that can be used as input to the LLAMA2 model.

    :param history: typing.List[str]: Store the history of user input and system response
    :param prompt: str: Specify the prompt to be displayed
    :param system: typing.Optional[str]: Indicate that the system is optional
    :param : Specify the system's response
    :return: A string that is a concatenation of the
    
    """
    do_strip = False
    if system is not None:
        texts = [f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n']
    else:
        texts = [f'<s>[INST] ']
    for user_input, response in history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    prompt = prompt.strip() if do_strip else prompt
    texts.append(f'{prompt} [/INST]')
    return "".join(texts)


def llama2_prompter_chat_format(
        system: str,
        messages: typing.List[str],
):
    """
    The llama2_prompter_chat_format function takes a system message and a list of messages,
    and returns the formatted string that can be used to create an LLAMA2 chat file.
    The system message is optional, and if it is not provided then the function will return only the user messages.
    The user messages are expected to be in pairs: one for each speaker (system or human).  The first element of each
     pair should be the name of that speaker.

    :param system: str: Store the system message
    :param messages: typing.List[str]: Pass in a list of strings
    :param : Add the system message to the beginning of the chat
    :return: A string that is the
    
    """
    if system is not None:
        string = [f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n']
    else:
        string = [f'<s>[INST] ']
    for index in range(0, len(messages), 2):
        string.append(
            f'{messages[index]} [/INST] {messages[index + 1].strip()} </s><s>[INST] ')
    return "".join(string).strip()
