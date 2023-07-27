# Copyright © 2023 Abacus.AI. All rights reserved.

def add_posembed_args(parser):
    parser.add_argument("--scale-context", type=float, help='Set context length scaling for interpolation.')
    parser.add_argument('--scale-power', type=float, help='Enable non-uniform frequency scaling.')
    parser.add_argument('--truncate-pos', type=int, default=0, help='The context length at which to truncate wavelengths.')
    parser.add_argument('--xpos', action='store_true', help='Use xpos embeddings.')
    parser.add_argument('--randomized', action='store_true', help='Enable randomized step sizes when scaling.')
    return parser


def load_model(base_model, delta_model, **patch_args):
    import torch
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_config(base_model, torch_dtype=torch.float16)
    delta_model = AutoModelForCausalLM.from_config(delta_model, torch_dtype=torch.float16)
    for name, param in base_model.named_parameters():
        delta_param = delta_model.get_parameter(name)
        assert delta_param.shape == param.shape
        delta_param += param

    from interpolate import ScaledLlamaRotaryEmbedding
    ScaledLlamaRotaryEmbedding.patch(delta_model, **patch_args)
    return delta_model
