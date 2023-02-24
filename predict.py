# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from cog import BasePredictor, BaseModel, Input
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig
from typing import List

from gddc.model import Model
from gddc.masking import get_masked_inputs


class Output(BaseModel):
    positions: List[List[int]]
    results: List[str]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model = Model(DistilBertModel(DistilBertConfig(31102)))
        model.load_state_dict(torch.load("./model.pt", map_location='cpu'))
        model.eval()
        self.model = model
        self.tokenizer: DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-german-cased')

    def predict(
        self,
        sentence: str = Input(description="Sentence to correct"),
    ) -> Output:
        """Run a single prediction on the model"""
        inputs, positions = get_masked_inputs(
            sentence, ['das', 'dass'], self.tokenizer.mask_token, lower=True)

        results = []
        # https://twitter.com/PyTorch/status/1437838231505096708
        # TODO: Replace with the compiled model if 2.0 is released
        with torch.inference_mode():
            for input in inputs:
                encoded = self.tokenizer(input, return_tensors='pt',
                                         padding='max_length', truncation=True, max_length=128)
                output: torch.Tensor = self.model(
                    encoded['input_ids'], encoded['attention_mask'])

                results.append(output.detach().item())

        # transform from tuples to lists
        positions = [[pos[0], pos[1]] for pos in positions]

        return Output(positions=positions, results=results)
