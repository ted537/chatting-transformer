from chattingtransformer import ChattingGPT2, greedy_decoding, beam_search

def test_beam_search():
    chatting = ChattingGPT2.from_pretrained(settings=beam_search(5))
    generated = chatting.generate_text(
        'Asia is a larger continent than Australia'
    )
    assert generated.startswith(
        "Asia is a larger continent than Australia, "
        "with a population of 1.5 billion people"
    )