# Copyright © 2023 Abacus.AI. All rights reserved.

def add_posembed_args(parser):
    parser.add_argument("--scale-context", type=float, help='Set context length scaling for interpolation.')
    parser.add_argument('--scale-power', type=float, help='Enable non-uniform frequency scaling.')
    parser.add_argument('--truncate-pos', type=int, default=0, help='The context length at which to truncate wavelengths.')
    parser.add_argument('--xpos', action='store_true', help='Use xpos embeddings.')
    parser.add_argument('--randomized', action='store_true', help='Enable randomized step sizes when scaling.')
    return parser


def load_model(base_model_path: str, delta_model_path: str = None, **patch_args):
    '''Helper to load a model and patch it to support a longer context.
    
    For example to load Giraffe V2 with its trained scale:
    ```python
    model = load_model('abacusai/Giraffe-v2-13b-32k', scale=8)
    ```

    To load a delta model you need the original llama v1 weights available:
    ```python
    model = load_model('abacusai/Giraffe-v1-delta-13b-scaled-4', 'path/to/llama-13b', scale=4)

    See `ScaledLlamaRotaryEmbedding.patch` for information on additional arguments.
    '''
    import torch
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_config(base_model_path, torch_dtype=torch.float16)
    if delta_model_path is not None:
        delta_model = AutoModelForCausalLM.from_config(delta_model_path, torch_dtype=torch.float16)
        for name, param in base_model.named_parameters():
            delta_param = delta_model.get_parameter(name)
            assert delta_param.shape == param.shape
            delta_param += param

    from interpolate import ScaledLlamaRotaryEmbedding
    ScaledLlamaRotaryEmbedding.patch(delta_model, **patch_args)
    return delta_model


def load_tokenizer(tokenizer_path = 'abacusai/Giraffe-v1-Tokenizer'):
    '''Load the tokenizer used for fine-tuning Giraffe models.
    For consistency, as of this time Giraffe models are fine-tuned with a Llama V1 tokenizer setup.
    Note: Future releases will upgrade to the Llama V2 tokenizer.
    '''
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
