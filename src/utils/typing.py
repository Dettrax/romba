"""Some useful type aliases relevant to this project."""

import pathlib
from typing import Literal, Sequence

import numpy
import torch
import transformers
import transformers.modeling_outputs

# `mamba-simple`, easier to understand the do different interventions
#from mamba_minimal.model import Mamba#, Mamba2LMHeadModel
from src.models import Mamba
# official implementation, superfast
# from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as Mamba

# huggingface implementation
# from transformers import MambaForCausalLM as Mamba

from typing import Union
import numpy as np
import torch

ArrayLike = Union[list, tuple, np.ndarray, torch.Tensor]
PathLike = Union[str, pathlib.Path]
Device = Union[str, torch.device]
# Throughout this codebase, we use HuggingFace model implementations.
# Model = (
#     transformers.GPT2LMHeadModel
#     | transformers.GPTJForCausalLM
#     | transformers.GPTNeoXForCausalLM
#     | transformers.LlamaForCausalLM
#     | Mamba
# )

from typing import Union

Model = Union[
    transformers.GPT2LMHeadModel,
    transformers.GPTJForCausalLM,
    transformers.GPTNeoXForCausalLM,
    transformers.LlamaForCausalLM,
    Mamba
]

ModelGenerateOutput = Union[
    transformers.generation.utils.GenerateOutput,
    torch.LongTensor
]

Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
#ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]
