import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from chattingtransformer.settings import POSSIBLE_METHODS, get_settings

# https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def eprint(*args,**kwargs):
    """Print to stderr so that user sees output but scripts don't consume it"""
    print(*args, file=sys.stderr, **kwargs)

class ChattingGPT2():
    tokenizer: PreTrainedTokenizerBase
    # model: 

    """
    # An easy to use class for state-of-the-art text generation.

    Key Features:
        1. Initialize a GPT2 model with just one line of code.
        2. Generate text in a single line of code with the your GPT2 model
        3. Select 1 of 4 GPT2 models.
        4. Fully customizable text generation parameters
    """

    valid_models = {
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl"
    }

    @staticmethod
    def create(model_name: str, model_type: str):
        self.tokenizer = 

    def __init__(self, model_name="gpt2"):

        if model_name not in self.valid_models:
            raise ValueError(f'"{model_name}" is not a valid model')

        eprint(f'Loading "{model_name}"')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pad_token_id = self.tokenizer.eos_token_id
        self._generation_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            pad_token_id=pad_token_id
        )
        eprint(f'Done loading "{model_name}"')


    def _assert_valid_generate_text(self, text: str, method: str) -> None:
        if not isinstance(text, str):
            raise TypeError('starting text must be a string')
        if len(text) == 0:
            raise ValueError('starting text must be non empty')
        if method not in POSSIBLE_METHODS:
            raise ValueError()

    def generate_text(self,
        starting_text: str, combine=True, method="top-k-sampling",
        custom_settings=None, min_length=20, max_length=60
    ) -> str:
        """
        :param: combine: if true, the starting text will be concatenated with the output.
        :param method: either one of 1/5 preconfigured methods, or "custom" to indicate custom settings
        :param custom_settings: if method == "custom", then custom settings may be provided in the form of
              a dictionary. Refer to the README to see potential parameters to add.
              Parameters that are not added to the dictionary will keep their default value.

        :return: Text that the model generates.
        """

        self._assert_valid_generate_text(starting_text)

        settings = get_settings(method, custom_settings, self.logger)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            do_sample=settings['do_sample'],
            early_stopping=settings['early_stopping'],
            num_beams=settings['num_beams'],
            temperature=settings['temperature'],
            top_k=settings['top_k'],
            top_p=settings['top_p'],
            repetition_penalty=settings['repetition_penalty'],
            length_penalty=settings['length_penalty'],
            no_repeat_ngram_size=settings['no_repeat_ngram_size'],
            bad_words_ids=settings['bad_words_ids'],
        )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        final_result = self.__gt_post_processing(result, text, combine)

    def __gt_post_processing(self, result, text, combine):
        """
        A method for processing the output of the model. More features will be added later.

        :param result: result the output of the model after being decoded
        :param text:  the original input to generate_text
        "parm combine: if true, returns  text and result concatenate together.
        :return: returns to text after going through  post-processing
        """

        if combine:
            return result

        return result[len(text):]

