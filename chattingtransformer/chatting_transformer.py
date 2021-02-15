import sys
from typing import Dict, Any
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from chattingtransformer.settings import POSSIBLE_METHODS, default_settings

# https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def eprint(*args,**kwargs):
    """Print to stderr so that user sees output but scripts don't consume it"""
    print(*args, file=sys.stderr, **kwargs)

VALID_MODELS = {
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl"
}

@dataclass
class ChattingGPT2:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    settings: Dict[str,Any]

    """
    # An easy to use class for state-of-the-art text generation.

    Key Features:
        1. Initialize a GPT2 model with just one line of code.
        2. Generate text in a single line of code with the your GPT2 model
        3. Select 1 of 4 GPT2 models.
        4. Fully customizable text generation parameters
    """

    @staticmethod
    def from_pretrained(model_name: str, model_type: str):
        if model_name not in VALID_MODELS:
            raise ValueError(f'"{model_name}" is not a valid model')

        eprint(f'Loading "{model_name}"')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            pad_token_id=tokenizer.eos_token_id
        )

        eprint(f'Done loading "{model_name}"')

        return ChattingGPT2(model, tokenizer, default_settings())

    def _assert_valid_generate_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError('starting text must be a string')
        if len(text) == 0:
            raise ValueError('starting text must be non empty')

    def generate_text(self,
        starting_text: str, include_starting_text=True,
        min_length=20, max_length=60
    ) -> str:
        """
        Generate text using a transformer model from some starting text
        """

        self._assert_valid_generate_text(starting_text)

        input_ids = self.tokenizer.encode(starting_text, return_tensors="pt")
        outputs = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            **self.settings
        )
        generated_tokens = outputs[0]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        


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

