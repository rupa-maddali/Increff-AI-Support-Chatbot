import chatbot, pytest

@pytest.mark.parametrize("q, expect", [
    ("Can I return a phone?", "30"),
    ("What cards can I pay with?", "Visa"),
    ("Is my data secure?", "TLS"),
])
def test_faq_lookup(q, expect):
    ans = chatbot.chat_once(q)
    assert expect.lower() in ans.lower()

def test_follow_up():
    chatbot.chat_once("Tell me about laptop chargers")
    ans = chatbot.chat_once("Will it work in Europe?")
    assert "adapter" in ans.lower()
