import random

import numpy as np


class ActionMasker:
    ACTION_INSTRUCTION = " Provide only space separated numbers"

    def __init__(
        self,
        tokenizer,
        mask_token="?",
    ):
        self.tokenizer = tokenizer
        self._mask_token_id = tokenizer.encode(mask_token, add_special_tokens=False)[0]
        self._action_sub_ids = tokenizer.encode(
            self.ACTION_INSTRUCTION, add_special_tokens=False
        )

    def __call__(self, inputs: dict, proba: float = 0.9, ratio=0.4):
        """Apply random masking to action messages in the input sequence.

        Args:
            inputs (dict): Dictionary containing:
                - input_ids: List[int], token IDs of input sequence
                - labels: List[int], corresponding labels (-100 for masked positions)
            proba (float): Probability [0-1] of applying masking (default: 0.9)
            ratio (float): Maximum ratio [0-1] of action tokens to mask (default: 0.4)

        Returns:
            dict: Modified inputs with:
                - Randomly masked action tokens replaced with mask_token_id
                - Corresponding labels set to -100 for masked positions

        Note:
            Only applies masking if input contains action message (checked via _check_match)
            Masking preserves the last EOS token in action sequence
        """
        # Add random mask to action message
        input_ids = np.array(inputs["input_ids"])
        labels = np.array(inputs["labels"])

        if not self._check_match(input_ids):
            return inputs

        action_indices = np.where(labels != -100)[0][:-1]  # remove last `eos` token
        ratio = random.uniform(0, ratio)
        mask_len = int(len(action_indices) * ratio)
        if random.random() >= proba or mask_len <= 0:
            return inputs

        # Create mask positions in action range
        mask_positions = random.sample(
            range(action_indices[0], action_indices[-1]), mask_len
        )
        labels[mask_positions] = -100
        # Replace the input_ids at mask_positions with mask_token_id(?)
        input_ids[mask_positions] = self._mask_token_id

        inputs["input_ids"] = input_ids.tolist()
        inputs["labels"] = labels.tolist()

        return inputs

    def _check_match(self, input_ids: np.ndarray):
        """Check if action message pattern(VLA0) exists in input sequence.

        Args:
            input_ids: 1D numpy array of token IDs

        Returns:
            bool: True if action pattern is found
        """
        n, m = len(input_ids), len(self._action_sub_ids)
        if m == 0 or n < m:
            return False

        # Create sliding windows of pattern length and compare with target pattern
        matches = np.all(
            np.lib.stride_tricks.sliding_window_view(input_ids, m)
            == self._action_sub_ids,
            axis=1,
        )
        return any(matches)
