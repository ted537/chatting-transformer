GenerateSettings = dict

"""
Degeneration methods
"""

def greedy_decoding() -> GenerateSettings:
    return {}

def beam_search(num_beams:int) -> GenerateSettings:
    return {
        'early_stopping': True,
        'num_beams':num_beams
    }

def generic_sampling(temperature=0.7) -> GenerateSettings:
    return {
        'do_sample': True,
        'temperature': temperature
    }

def top_k_sampling(k:int=50, temperature:float=0.7) -> GenerateSettings:
    return {
        'do_sample':True,
        'top_k': k,
        'temperature': temperature
    }

def top_p_sampling(p:float=0.92, temperature:float=0.7) -> GenerateSettings:
    """Also known as nucleus sampling"""
    return {
        'do_sample':True,
        'top_p':p,
        'temperature':temperature
    }

default_settings = greedy_decoding