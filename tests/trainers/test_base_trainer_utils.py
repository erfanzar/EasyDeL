from easydel.trainers.base_trainer import BaseTrainer


def test_normalize_prompts_plain_string():
    normalized = BaseTrainer._normalize_esurge_prompts("hello", apply_chat_template=False)
    assert normalized == ["hello"]


def test_normalize_prompts_chat_wrapping():
    normalized = BaseTrainer._normalize_esurge_prompts("hello", apply_chat_template=True)
    assert len(normalized) == 1
    convo = normalized[0]
    assert isinstance(convo, list)
    assert convo[0]["role"] == "user"
    assert convo[0]["content"] == "hello"


def test_normalize_prompts_double_wrapped_chat_passes_through():
    chat = [[{"role": "user", "content": "hi"}]]
    normalized = BaseTrainer._normalize_esurge_prompts(chat, apply_chat_template=False)
    assert normalized == chat


def test_normalize_prompts_list_of_strings():
    prompts = ["first", "second"]
    normalized = BaseTrainer._normalize_esurge_prompts(prompts, apply_chat_template=False)
    assert normalized == prompts
