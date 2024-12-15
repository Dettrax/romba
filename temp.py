import sys
import torch
import transformers

from src import functional

import logging
from src import models

from src.utils import logging_utils
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format=logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
    stream=sys.stdout,
)

torch.__version__, transformers.__version__, torch.version.cuda

from src.models import ModelandTokenizer

mt = ModelandTokenizer(
    torch_dtype=torch.float32
)

#####################################################
subject = "USA"
prompt_template = "The president of {} is"

prompt = prompt_template.format(subject)
prompt

from src.functional import predict_next_token

predict_next_token(
    mt,
    # prompt=prompt,
    prompt = prompt_template.format("USA"),
    k=5,
)

# Update the request with the target change
request = {
    "prompt": prompt_template,
    "subject": subject,
    "target_new": {"str": " Modi"},  # Note the space before Modi - important for tokenization
}

# Expanded generation prompts to test both scenarios
generation_prompts = [
    # Next token predictions
    f"The president of {subject} is",
    f"Who leads {subject}? The president is",
    f"{subject}'s president is",

    # Mid-sequence predictions
    f"The president of {subject} was seen giving a speech",
    f"I heard the president of {subject} just announced",
    f"A meeting with {subject}'s president discussed",

    # Control prompts
    f"The president of India is",
    f"The president of Russia is",

    # Test different positions
    f"Yesterday, the president of {subject} made an announcement",
    f"Breaking news: {subject}'s president has just"
]

# Expanded context templates to capture various positions
context_templates = [
    '{}',
    'Latest update about {}',
    'In a recent development, {}',
    'Sources confirm that {}',
    'Reports indicate {}',
    'According to officials, {}',
    'Breaking: {}',
    'Update regarding {}'
]

words = [subject] * len(context_templates)


from src.rome_utils import nethook

tokenized = mt.tokenizer(prompt, return_tensors="pt", padding=True, return_offsets_mapping=True).to(mt.device)
offsets = tokenized.pop("offset_mapping")

[(idx, mt.tokenizer.decode(t)) for idx, t in enumerate(tokenized.input_ids[0])]

from src.rome.rome_hparams import ROMEHyperParams

hparams = ROMEHyperParams(
    layers=list(range(7,15,2)),
    fact_token="subject_last",
    v_num_grad_steps=50,
    v_lr=5e-1,
    v_loss_layer=models.determine_layers(mt)[-1],
    v_weight_decay=0.5,
    clamp_norm_factor=3,
    kl_factor=0.0325,
    mom2_adjustment=True,
    context_template_length_params=[[5, 10], [10, 10], [15, 5]],  # Added longer context

    rewrite_module_tmp=mt.layer_name_format + ".mixer.in_proj",
    layer_module_tmp=mt.layer_name_format,
    mlp_module_tmp="",
    attn_module_tmp="",
    ln_f_module=models.determine_final_layer_norm_path(mt),
    lm_head_module=models.determine_lm_head_path(mt),

    mom2_dataset="wikipedia",
    mom2_n_samples=1000,
    mom2_dtype="float32",

    mamba_block_non_ssm=True,  # will effect the non-ssm flow only, default is false
    # mamba_block_ssm=True, # will effect the ssm flow only, default is false
)

import json

print(json.dumps(hparams.__dict__, indent=2))

from src.rome.compute_v import compute_v


functional.free_gpu_cache()

from src.rome.rome_main import (
    apply_rome_to_model,
    restore_weights,
    save_weights,
)

# Create multiple requests
requests = [
    {
        "prompt": "The president of {} is",
        "subject": "USA",
        "target_new": {"str": " Modi"}
    },
    {
        "prompt": "The president of {} recently",
        "subject": "USA",
        "target_new": {"str": " Modi"}
    }
]

# Apply ROME sequentially for each prompt
for request in requests:
    model, orig_weights = apply_rome_to_model(
        mt=mt,
        requests=request,
        hparams=hparams
    )


rome_weights = save_weights(model, list(orig_weights.keys()))

from src.utils.generation import generate_fast

restore_weights(model, rome_weights)
generate_fast(
    mt = mt,
    prompts = generation_prompts,
    max_out_len = 10,
)

restore_weights(model, orig_weights)
generate_fast(
    mt = mt,
    prompts = generation_prompts,
    max_out_len = 10,
)

