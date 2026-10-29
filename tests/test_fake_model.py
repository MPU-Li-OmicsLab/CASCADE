# tests/test_fake_model.py
from cascade.models.base import ChatModel
from cascade.models.fake import FakeChatModel


def test_fake_returns_scripted_in_order():
    m = FakeChatModel(model_id="fake-1", scripts=["hi", "there"])
    assert m.model_id == "fake-1"
    assert m.generate([{"role": "user", "content": "a"}]) == "hi"
    assert m.generate([{"role": "user", "content": "b"}]) == "there"


def test_fake_repeats_last_when_exhausted():
    m = FakeChatModel(model_id="f", scripts=["only"])
    assert m.generate([]) == "only"
    assert m.generate([]) == "only"  # repeats last instead of crashing


def test_fake_records_calls():
    m = FakeChatModel(model_id="f", scripts=["x"])
    m.generate([{"role": "user", "content": "hello"}])
    assert m.calls[-1][0]["content"] == "hello"


def test_fake_is_a_chatmodel():
    assert isinstance(FakeChatModel(model_id="f", scripts=["x"]), ChatModel)
