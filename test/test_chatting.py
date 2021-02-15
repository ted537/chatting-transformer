from chattingtransformer import (
    ChattingGPT2, 
    greedy_decoding, beam_search, generic_sampling,
    top_k_sampling, top_p_sampling
)


def test_greedy_decoding():
    chatting = ChattingGPT2.from_pretrained('gpt2','gpt2',greedy_decoding())
    generated = chatting.generate_text(
        'Asia is a larger continent than Australia'
    )
    assert generated.startswith(
        "Asia is a larger continent than Australia, "
        "and the United States is a smaller continent than Australia"
    )


def test_beam_search():
    chatting = ChattingGPT2.from_pretrained('gpt2','gpt2',beam_search(5))
    generated = chatting.generate_text(
        'Asia is a larger continent than Australia'
    )
    assert generated.startswith(
        "Asia is a larger continent than Australia, "
        "with a population of 1.5 billion people"
    )


def test_sampling_diversity():
    chatting = ChattingGPT2.from_pretrained('gpt2','gpt2',generic_sampling())
    STARTING_TEXT = "Asia is a larger continent than Australia"
    first_generation = chatting.generate_text(STARTING_TEXT)
    second_generation = chatting.generate_text(STARTING_TEXT)
    assert first_generation != second_generation


def test_all_methods_work():
    ALL_SETTINGS = [
        greedy_decoding(), beam_search(),
        generic_sampling(),
        top_k_sampling(), top_p_sampling()
    ]
    for settings in ALL_SETTINGS:
        chatting = ChattingGPT2.from_pretrained('gpt2','gpt2',settings)
        chatting.generate_text("Asia is a larger continent than Austrlia")