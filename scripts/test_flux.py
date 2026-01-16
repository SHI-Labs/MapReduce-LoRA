import torch
from diffusers import FluxPipeline
from peft import PeftModel
import os
import sys
import random
import argparse
import json
import csv
import numpy as np

# add project root to Python path so `flow_grpo` imports work from scripts/debug
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob

# global seed to mirror train eval seeding before any resize_token_embeddings
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def ensure_control_token_in_prompts(prompts, token):
    if not token:
        return prompts
    return [p if token in p else (p + " " + token) for p in prompts]


def parse_token_spec(spec: str):
    if not spec:
        return [], {}
    toks = spec.strip().split()
    order = []
    counts = {}
    i = 0
    last_token = None
    while i < len(toks):
        part = toks[i]
        if (":" in part) or ("=" in part):
            key, val = (part.split(":", 1) if ":" in part else part.split("=", 1))
            key, val = key.strip(), val.strip()
            if key and key not in order:
                order.append(key)
            if key and val.replace('.', '', 1).isdigit():
                try:
                    counts[key] = max(1, int(float(val)))
                except Exception:
                    pass
            last_token = key
            i += 1
        else:
            token = part
            if token not in order:
                order.append(token)
            last_token = token
            if i + 1 < len(toks) and toks[i + 1].replace('.', '', 1).isdigit():
                try:
                    counts[token] = max(1, int(float(toks[i + 1])))
                except Exception:
                    pass
                i += 2
            else:
                i += 1
    for t in order:
        counts.setdefault(t, 1)
    return order, counts


def parse_token_weights(spec: str):
    if not spec:
        return {}
    toks = spec.strip().split()
    result = {}
    i = 0
    while i < len(toks):
        part = toks[i]
        if (":" in part) or ("=" in part):
            key, val = (part.split(":", 1) if ":" in part else part.split("=", 1))
            key, val = key.strip(), val.strip()
            try:
                result[key] = float(val)
            except Exception:
                pass
            i += 1
        else:
            if i + 1 < len(toks):
                try:
                    result[part] = float(toks[i + 1])
                    i += 2
                    continue
                except Exception:
                    pass
            i += 1
    return result


def ensure_control_tokens_with_strategy(prompts, available_tokens, strategy="all", token_order=None, token_counts=None):
    if not available_tokens:
        return prompts

    token_counts = token_counts or {}
    result = []
    for prompt in prompts:
        existing_tokens = [token for token in available_tokens if token in prompt]
        temp_prompt = prompt
        for token in existing_tokens:
            temp_prompt = temp_prompt.replace(token, "").strip()
        temp_prompt = " ".join(temp_prompt.split())

        if token_order:
            final_tokens = [token for token in token_order if token in available_tokens]
        else:
            final_tokens = available_tokens

        expanded_tokens = []
        for t in final_tokens:
            n = int(max(1, token_counts.get(t, 1)))
            expanded_tokens.extend([t] * n)

        if strategy == "all":
            prompt = temp_prompt + " " + " ".join(expanded_tokens)
        elif strategy == "first":
            first_t = expanded_tokens[0] if expanded_tokens else final_tokens[0]
            prompt = temp_prompt + " " + first_t
        elif strategy == "smart":
            chosen_token = choose_token_by_content(temp_prompt, final_tokens)
            n = int(max(1, token_counts.get(chosen_token, 1)))
            prompt = temp_prompt + " " + " ".join([chosen_token] * n)

        result.append(prompt)

    return result


def choose_token_by_content(prompt, available_tokens):
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ['text', 'word', 'letter', 'sign', 'read']):
        if '<ocr>' in available_tokens:
            return '<ocr>'
    if any(word in prompt_lower for word in ['quality', 'beautiful', 'aesthetic', 'good', 'best']):
        if '<picks>' in available_tokens:
            return '<picks>'
    return available_tokens[0]


class MultiEmbeddingManager:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers
        self.control_tokens = {}
        self.enabled = True
        self.token_scales = {}

    def add_control_token(self, control_token, embedding_path=None, encoders=None):
        if control_token in self.control_tokens:
            print(f"Control token {control_token} already exists, replacing...")
            self._remove_hooks(control_token)

        self.control_tokens[control_token] = {
            'token_ids': [],
            'params': [],
            'hooks': []
        }
        self._ensure_token_exists(control_token, encoders)

        if embedding_path and os.path.exists(embedding_path):
            state_dict = torch.load(embedding_path, map_location="cpu")
            self._load_token_embedding(control_token, state_dict)
            print(f"Loaded embedding for {control_token} from {embedding_path}")
        elif embedding_path:
            print(f"Warning: No embedding file found for {control_token} at {embedding_path}")

    def _ensure_token_exists(self, control_token, encoders=None):
        token_data = self.control_tokens[control_token]
        token_data['token_ids'] = []
        token_data['params'] = []
        for idx, tok in enumerate(self.tokenizers):
            if control_token not in tok.get_vocab():
                tok.add_tokens([control_token])
                if encoders and idx < len(encoders):
                    try:
                        encoders[idx].resize_token_embeddings(len(tok))
                    except Exception:
                        pass
            tid = tok.convert_tokens_to_ids(control_token)
            token_data['token_ids'].append(tid)
            if encoders and idx < len(encoders):
                emb = encoders[idx].get_input_embeddings()
                init_vec = emb.weight[tid].detach().clone().to(dtype=torch.float32)
            else:
                init_vec = torch.randn(4096, dtype=torch.float32) * 0.02
            token_data['params'].append(torch.nn.Parameter(init_vec))

    def _load_token_embedding(self, control_token, state_dict):
        token_data = self.control_tokens[control_token]
        for i, param in enumerate(token_data['params']):
            key = f"encoder_{i}"
            if key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict[key].to(device=param.device, dtype=param.dtype))
                print(f"Loaded encoder_{i} embedding for {control_token}")
            else:
                print(f"Warning: No embedding found for encoder_{i} in {control_token}")

    def attach(self, encoders):
        for control_token in self.control_tokens:
            self._remove_hooks(control_token)
        for idx, (tok, enc) in enumerate(zip(self.tokenizers, encoders)):
            try:
                enc.resize_token_embeddings(len(tok))
            except Exception:
                pass
        for control_token, token_data in self.control_tokens.items():
            self._attach_token_hooks(control_token, encoders)

    def _attach_token_hooks(self, control_token, encoders):
        token_data = self.control_tokens[control_token]
        for tid, enc, param in zip(token_data['token_ids'], encoders, token_data['params']):
            emb = enc.get_input_embeddings()
            if param.device != emb.weight.device:
                with torch.no_grad():
                    param.data = param.data.to(device=emb.weight.device)

            def _hook(module, inputs, output, tid=tid, param=param, token_name=control_token, self_ref=self):
                if not self_ref.enabled:
                    return output
                input_ids = inputs[0]
                if not torch.is_floating_point(output):
                    return output
                mask = (input_ids == tid).unsqueeze(-1)
                if mask.any():
                    scale = float(self_ref.token_scales.get(token_name, 1.0))
                    expanded = (param.view(1, 1, -1) * scale).expand(
                        output.shape[0], output.shape[1], output.shape[2]
                    ).to(dtype=output.dtype)
                    return torch.where(mask.expand_as(output), expanded, output)
                return output

            hook = emb.register_forward_hook(_hook)
            token_data['hooks'].append(hook)

    def _remove_hooks(self, control_token):
        if control_token in self.control_tokens:
            for hook in self.control_tokens[control_token]['hooks']:
                try:
                    hook.remove()
                except Exception:
                    pass
            self.control_tokens[control_token]['hooks'] = []

    def get_token_ids(self, control_token):
        if control_token in self.control_tokens:
            return self.control_tokens[control_token]['token_ids']
        return []

    def list_tokens(self):
        return list(self.control_tokens.keys())

    def load_state_dict(self, state_dict, control_token, strict: bool = False):
        if control_token in self.control_tokens:
            self._load_token_embedding(control_token, state_dict)
        else:
            print(f"Warning: Control token {control_token} not found in manager")


class EmbeddingManager:
    def __init__(self, tokenizers, control_token):
        self.tokenizers = tokenizers
        self.control_token = control_token
        self.token_ids = []
        self.params = []
        self.hooks = []
        self.enabled = True
        self.token_scale = 1.0

    def ensure_token_exists(self, encoders):
        self.token_ids = []
        self.params = []
        for idx, (tok, enc) in enumerate(zip(self.tokenizers, encoders)):
            if self.control_token not in tok.get_vocab():
                tok.add_tokens([self.control_token])
                try:
                    enc.resize_token_embeddings(len(tok))
                except Exception:
                    pass
            tid = tok.convert_tokens_to_ids(self.control_token)
            self.token_ids.append(tid)
            emb = enc.get_input_embeddings()
            init_vec = emb.weight[tid].detach().clone().to(dtype=torch.float32)
            self.params.append(torch.nn.Parameter(init_vec))

    def attach(self, encoders):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []
        for tid, enc, param in zip(self.token_ids, encoders, self.params):
            emb = enc.get_input_embeddings()
            if param.device != emb.weight.device:
                with torch.no_grad():
                    param.data = param.data.to(device=emb.weight.device)

            def _hook(module, inputs, output, tid=tid, param=param, self_ref=self):
                if not self_ref.enabled:
                    return output
                input_ids = inputs[0]
                if not torch.is_floating_point(output):
                    return output
                mask = (input_ids == tid).unsqueeze(-1)
                if mask.any():
                    expanded = (param.view(1, 1, -1) * float(self_ref.token_scale)).expand(
                        output.shape[0], output.shape[1], output.shape[2]
                    ).to(dtype=output.dtype)
                    return torch.where(mask.expand_as(output), expanded, output)
                return output

            self.hooks.append(emb.register_forward_hook(_hook))

    def load_state_dict(self, state_dict, strict: bool = False):
        for i, p in enumerate(self.params):
            key = f"encoder_{i}"
            if key in state_dict:
                with torch.no_grad():
                    p.copy_(state_dict[key].to(device=p.device, dtype=p.dtype))
        return None


def average_lora_weights(lora_paths, base_model):
    if len(lora_paths) == 1:
        print(f"Loading single LoRA from: {lora_paths[0]}")
        model = PeftModel.from_pretrained(base_model, lora_paths[0])
        return model

    print(f"Averaging {len(lora_paths)} LoRA weights...")
    print(f"Loading base LoRA from: {lora_paths[0]}")
    model = PeftModel.from_pretrained(base_model, lora_paths[0])
    base_state_dict = model.state_dict()
    all_state_dicts = [base_state_dict]
    for i, lora_path in enumerate(lora_paths[1:], 1):
        print(f"Loading LoRA {i+1}/{len(lora_paths)} from: {lora_path}")
        temp_model = PeftModel.from_pretrained(base_model, lora_path)
        all_state_dicts.append(temp_model.state_dict())
        del temp_model
    print("Computing average weights...")
    averaged_state_dict = {}
    for key in base_state_dict.keys():
        if any('lora' in key.lower() for part in key.split('.')):
            stacked_tensors = torch.stack([sd[key] for sd in all_state_dicts])
            averaged_state_dict[key] = torch.mean(stacked_tensors, dim=0)
        else:
            averaged_state_dict[key] = base_state_dict[key]
    model.load_state_dict(averaged_state_dict)
    print("Successfully averaged LoRA weights")
    return model


def load_prompts(prompts_file):
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found!")
        return []
    print(f"Loading prompts from: {prompts_file}")
    if "geneval" in str(prompts_file):
        with open(prompts_file) as fp:
            metadatas = [json.loads(line) for line in fp]
            return metadatas
    else:
        with open(prompts_file, "r", encoding="utf-8") as f:
            if "GenAI-Bench" in str(prompts_file):
                prompts_data = json.load(f)
                prompts = []
                for key, entry in prompts_data.items():
                    prompts.append((key, entry))
                print(f"Loaded {len(prompts)} prompts from GenAI-Bench format")
                return prompts
            else:
                prompts = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(prompts)} prompts from text format")
                return prompts


def process_prompt_item(prompt_item, index, pipe_flux, results_dir, add_control_token, control_token, num_steps: int, guidance_scale: float, args=None):
    if isinstance(prompt_item, tuple):
        key, entry = prompt_item
        prompt = entry['prompt']
        seed = entry["random_seed"]
        print(f"\n--- Processing GenAI-Bench prompt {key}: {prompt} | seed: {seed}")
    else:
        prompt = prompt_item
        seed = 42
        print(f"\n--- Processing prompt {index:06d}: {prompt} | seed: {seed}")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    if add_control_token:
        if hasattr(args, '_loaded_control_tokens') and args._loaded_control_tokens:
            prompt = ensure_control_tokens_with_strategy(
                [prompt],
                args._loaded_control_tokens,
                args.token_strategy,
                getattr(args, '_token_order', None),
                getattr(args, '_token_counts', None)
            )[0]
        else:
            prompt = ensure_control_token_in_prompts([prompt], control_token)[0]

    print(f"Prompt: {prompt}")

    prompt_embeds, pooled_prompt_embeds, _ = encode_prompt(
        [pipe_flux.text_encoder, pipe_flux.text_encoder_2],
        [pipe_flux.tokenizer, pipe_flux.tokenizer_2],
        prompt,
        max_sequence_length=128,
    )

    with torch.no_grad():
        images_base, _, _, _, _ = pipeline_with_logprob(
            pipe_flux,
            prompt_embeds=prompt_embeds.to(pipe_flux.device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(pipe_flux.device),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            output_type="pil",
            height=512,
            width=512,
            noise_level=0,
            generator=generator,
        )

    if "geneval" in results_dir:
        image_path = os.path.join(results_dir, f"{index:05d}/samples/{index:05d}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    elif isinstance(prompt_item, tuple):
        image_path = os.path.join(results_dir, f"{key}.png")
    else:
        image_path = os.path.join(results_dir, f"{index:06d}.png")

    images_base[0].save(image_path)
    print(f"Image saved: {image_path} | prompt: {prompt}")

    return {
        "prompt": prompt,
        "seed": seed,
        "image_path": image_path
    }


def process_prompt_batch(prompts_batch, save_indices, pipe_flux, results_dir, add_control_token, control_token, num_steps: int, guidance_scale: float, args=None):
    if add_control_token:
        if hasattr(args, '_loaded_control_tokens') and args._loaded_control_tokens:
            prompts_batch = ensure_control_tokens_with_strategy(
                prompts_batch,
                args._loaded_control_tokens,
                args.token_strategy,
                getattr(args, '_token_order', None),
                getattr(args, '_token_counts', None)
            )
        else:
            prompts_batch = ensure_control_token_in_prompts(prompts_batch, control_token)

    prompt_embeds, pooled_prompt_embeds, _ = encode_prompt(
        [pipe_flux.text_encoder, pipe_flux.text_encoder_2],
        [pipe_flux.tokenizer, pipe_flux.tokenizer_2],
        prompts_batch,
        max_sequence_length=128,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(1)

    with torch.no_grad():
        images_base, _, _, _, _ = pipeline_with_logprob(
            pipe_flux,
            prompt_embeds=prompt_embeds.to(pipe_flux.device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(pipe_flux.device),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            output_type="pil",
            height=512,
            width=512,
            noise_level=0,
            generator=generator,
        )

    for offset, img in enumerate(images_base):
        global_index = save_indices[offset]
        if "geneval" in results_dir:
            image_path = os.path.join(results_dir, f"{global_index:05d}/samples/{global_index:05d}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        else:
            image_path = os.path.join(results_dir, f"{global_index:06d}.png")
        img.save(image_path)
        print(f"Image saved: {image_path} | prompt: {prompts_batch[offset]}")


def main():
    parser = argparse.ArgumentParser(description='Generate images using FLUX with optional LoRA fine-tuning')

    parser.add_argument('--prompts_file', type=str,
                        default=None,
                        help='Path to the prompts file (default: ../data/GenAI-Bench/genai_image_seed.json)')
    parser.add_argument('--prompt', type=str, 
                       default=None,
                       help='given prompt to generate image')
    parser.add_argument('--results_dir', type=str, default="results/",
                        help='Directory to save results')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                        help='Directory to save results')

    parser.add_argument('--lora_checkpoints', type=str, nargs='+', default=None,
                        help='Paths to multiple LoRA checkpoint directories for averaging. Usage: --lora_checkpoints path1 path2 path3')
    parser.add_argument('--average_loras', action='store_true',
                        help='Enable averaging of multiple LoRA weights. Requires --lora_checkpoints with multiple paths')

    parser.add_argument('--mode', type=str, default="eval_geneval",
                        help='Mode: eval_geneval, eval_genaibench, eval_pickscore, eval_ocr, eval_partiprompt')
    parser.add_argument('--use_adapter', action='store_true',
                        help='Use adapter if flag is present')
    parser.add_argument('--add_control_token', action='store_true',
                        help='Add control token if flag is present')

    parser.add_argument('--ti_emb_path', type=str, default="",
                        help='Path to TI embeddings')
    parser.add_argument('--control_token', type=str, default=None,
                        help='Control token: <alt>, <ocr>, <picks>')
    parser.add_argument('--all_control_tokens', type=str, nargs='+', default=["<alt>", "<ocr>", "<picks>"],
                        help='All control tokens. Usage: --all_control_tokens "<alt>" "<ocr>" "<picks>" (use quotes in bash)')

    parser.add_argument('--alt_emb_path', type=str, default="",
                        help='Path to <alt> token embeddings')
    parser.add_argument('--ocr_emb_path', type=str, default="",
                        help='Path to <ocr> token embeddings')
    parser.add_argument('--picks_emb_path', type=str, default="",
                        help='Path to <picks> token embeddings')

    parser.add_argument('--embedding_mode', type=str, default='auto', choices=['single', 'multi', 'auto'],
                        help='Embedding loading mode: single (use ti_emb_path + control_token), multi (use specific paths), auto (detect based on provided paths)')

    parser.add_argument('--token_strategy', type=str, default='all', choices=['first', 'all', 'smart'],
                        help='Strategy for control tokens')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of prompts to process per forward pass')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of shards for multi-GPU/process inference')
    parser.add_argument('--shard_idx', type=int, default=0,
                        help='This process shard index [0, num_shards)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                        help='Inference dtype')
    parser.add_argument('--enable_xformers', action='store_true',
                        help='Enable xFormers memory efficient attention if available')
    parser.add_argument('--enable_sdpa', action='store_true',
                        help='Prefer PyTorch SDPA kernels when available')
    parser.add_argument('--steps', type=int, default=28,
                        help='num_inference_steps')
    parser.add_argument('--guidance', type=float, default=3.5,
                        help='guidance_scale; set 1.0 to disable CFG')

    parser.add_argument('--token_spec', type=str, default=os.environ.get('TOKEN_SPEC', ''),
                        help='Combined order+counts, e.g., "<picks> 3 <ocr> 2 <alt> 2" or "<ocr> <picks> <alt>"')
    parser.add_argument('--token_weights', type=str, default=os.environ.get('TOKEN_WEIGHTS', ''),
                        help='Per-token weights, e.g. "<alt>:1.5 <ocr>:0.8" (scales embeddings)')

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # set hf_cache_dir for loading model from local
    from pathlib import Path

    def get_hf_hub_cache_dir() -> Path:
        if (v := os.environ.get("HUGGINGFACE_HUB_CACHE")):
            return Path(v)
        if (v := os.environ.get("HF_HOME")):
            return Path(v) / "hub"
        xdg = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg) / "huggingface" / "hub"
    # print(get_hf_hub_cache_dir())

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    flux_dtype = dtype_map[args.dtype]

    try:
        print("Attempting to load from local path...")
        pipe = FluxPipeline.from_pretrained(get_hf_hub_cache_dir() / "models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
        print("Successfully loaded from local path")
    except Exception as e:
        print(f"Failed to load from local path: {e}")
        print("Falling back to HuggingFace Hub...")
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        print("Successfully loaded from HuggingFace Hub")
    pipe = pipe.to("cuda")
    try:
        pipe.safety_checker = None
    except Exception:
        pass

    # Dtype alignment (keep VAE in fp32 like training script)
    pipe.to(dtype=torch.float32)
    # pipe.text_encoder.to(dtype=flux_dtype)
    # pipe.text_encoder_2.to(dtype=flux_dtype)
    # pipe.transformer.to(dtype=flux_dtype, device="cuda")

    if args.use_adapter:
        if args.average_loras and args.lora_checkpoints and len(args.lora_checkpoints) > 1:
            print(f"Loading and averaging {len(args.lora_checkpoints)} LoRA weights for transformer")
            missing_paths = [path for path in args.lora_checkpoints if not os.path.exists(path)]
            if missing_paths:
                raise ValueError(f"LoRA checkpoint paths not found: {missing_paths}")
            pipe.transformer = average_lora_weights(args.lora_checkpoints, pipe.transformer)
            pipe.transformer.set_adapter("default")
        elif args.average_loras and args.lora_checkpoints and len(args.lora_checkpoints) == 1:
            print("Warning: average_loras flag set but only one LoRA path provided. Using single LoRA.")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_checkpoints[0])
            pipe.transformer.set_adapter("default")
        elif args.average_loras:
            raise ValueError("average_loras flag requires --lora_checkpoints with multiple paths")
        else:
            if args.lora_checkpoint:
                lora_path = args.lora_checkpoint
                if os.path.isdir(os.path.join(lora_path, "lora")):
                    lora_path = os.path.join(lora_path, "lora")
                print("Loading pre-trained lora weights for transformer")
                pipe.transformer = PeftModel.from_pretrained(
                    pipe.transformer,
                    lora_path
                )
                pipe.transformer.set_adapter("default")

        try:
            print("Merging LoRA into base weights for faster inference...")
            pipe.transformer = pipe.transformer.merge_and_unload()
            pipe.transformer.to(device="cuda")#, dtype=flux_dtype)
        except Exception as e:
            print(f"Warning: could not merge LoRA into base weights: {e}")

    if args.add_control_token:
        print("pre-registering all control tokens")
        all_ctrl_tokens = list(dict.fromkeys(getattr(args, "all_control_tokens", [])))
        if all_ctrl_tokens:
            for tok, enc in zip([pipe.tokenizer, pipe.tokenizer_2], [pipe.text_encoder, pipe.text_encoder_2]):
                to_add = [t for t in all_ctrl_tokens if t not in tok.get_vocab()]
                if to_add:
                    tok.add_tokens(to_add)
                    try:
                        enc.resize_token_embeddings(len(tok))
                    except Exception:
                        pass
            ids = [pipe.tokenizer.convert_tokens_to_ids(t) for t in all_ctrl_tokens]
            print(f"Registered control tokens (tok0): {list(zip(all_ctrl_tokens, ids))}")

        embedding_paths = {
            "<alt>": args.alt_emb_path,
            "<ocr>": args.ocr_emb_path,
            "<picks>": args.picks_emb_path
        }
        valid_embedding_paths = {k: v for k, v in embedding_paths.items() if v and os.path.exists(v)}
        print(f"Valid embedding paths: {valid_embedding_paths}")

        if args.embedding_mode == 'auto':
            if len(valid_embedding_paths) > 1:
                embedding_mode = 'multi'
            elif args.ti_emb_path and args.control_token:
                embedding_mode = 'single'
            elif len(valid_embedding_paths) == 1:
                embedding_mode = 'multi'
            else:
                embedding_mode = 'single'
        else:
            embedding_mode = args.embedding_mode
        print(f"Using embedding mode: {embedding_mode}")

        if embedding_mode == 'multi':
            mgr = MultiEmbeddingManager([pipe.tokenizer, pipe.tokenizer_2])
            for token, path in valid_embedding_paths.items():
                mgr.add_control_token(token, path, [pipe.text_encoder, pipe.text_encoder_2])
            if not valid_embedding_paths and args.ti_emb_path and args.control_token:
                mgr.add_control_token(args.control_token, args.ti_emb_path, [pipe.text_encoder, pipe.text_encoder_2])

            token_weights = parse_token_weights(getattr(args, 'token_weights', ''))
            if token_weights:
                for t, w in token_weights.items():
                    mgr.token_scales[t] = float(w)

            mgr.attach([pipe.text_encoder, pipe.text_encoder_2])
            loaded_tokens = mgr.list_tokens()
            print(f"Loaded control tokens: {loaded_tokens}")

            args._loaded_control_tokens = loaded_tokens
            spec_order, spec_counts = parse_token_spec(getattr(args, 'token_spec', ''))
            args._token_order = spec_order if spec_order else None
            args._token_counts = spec_counts
        else:
            if not args.control_token:
                raise ValueError("control_token must be specified for single embedding mode")
            mgr = EmbeddingManager([pipe.tokenizer, pipe.tokenizer_2], args.control_token)
            mgr.ensure_token_exists([pipe.text_encoder, pipe.text_encoder_2])

            token_weights = parse_token_weights(getattr(args, 'token_weights', ''))
            if token_weights and args.control_token in token_weights:
                mgr.token_scale = float(token_weights[args.control_token])

            mgr.attach([pipe.text_encoder, pipe.text_encoder_2])

            print(f"Loading TI embeddings for {args.control_token}")
            if args.ti_emb_path and os.path.exists(args.ti_emb_path):
                mgr.load_state_dict(torch.load(args.ti_emb_path, map_location="cpu"))
                args._loaded_control_tokens = [args.control_token]
            else:
                print(f"Warning: No embedding file found at {args.ti_emb_path}")
                args._loaded_control_tokens = []

    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if getattr(args, 'enable_sdpa', False):
        try:
            if hasattr(torch.backends.cuda, 'sdp_kernel'):
                torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass

    if getattr(args, 'enable_xformers', False):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory efficient attention")
        except Exception as e:
            print(f"Failed to enable xFormers: {e}")

    if args.mode == "eval_geneval":
        if args.prompts_file is None:
            args.prompts_file = "./geneval/prompts/evaluation_metadata.jsonl"
            print("Using default prompts file: ./geneval/prompts/evaluation_metadata.jsonl")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return

        print(f"Processing {len(prompts)} prompts...")

        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch = [prompts[i] for i in current_indices]
            batch_texts = [(item["prompt"] if isinstance(item, dict) and "prompt" in item else item) for item in batch]
            process_prompt_batch(batch_texts, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
            if "geneval" in args.results_dir:
                for gi, item in zip(current_indices, batch):
                    meta_path = os.path.join(args.results_dir, f"{gi:05d}", "metadata.jsonl")
                    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                    with open(meta_path, "w") as fp:
                        json.dump(item, fp)
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")

    elif args.mode == "eval_genaibench":
        if args.prompts_file is None:
            args.prompts_file = "./data/GenAI-Bench/genai_image_seed.json"
            print("Using default prompts file: ./data/GenAI-Bench/genai_image_seed.json")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return

        print(f"Processing {len(prompts)} prompts...")

        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch = [prompts[i] for i in current_indices]
            batch_texts = []
            for item in batch:
                if isinstance(item, tuple):
                    _, entry = item
                    if isinstance(entry, dict):
                        batch_texts.append(entry.get("prompt", ""))
                    else:
                        batch_texts.append(str(entry))
                elif isinstance(item, dict):
                    batch_texts.append(item.get("prompt", ""))
                else:
                    batch_texts.append(str(item))
            process_prompt_batch(batch_texts, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")

    elif args.mode == "eval_pickscore":
        if args.prompts_file is None:
            args.prompts_file = "./dataset/pickscore/test.txt"
            print("Using default prompts file: ./dataset/pickscore/test.txt")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return

        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch = [prompts[i] for i in current_indices]
            process_prompt_batch(batch, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")

    elif args.mode == "eval_ocr":
        if args.prompts_file is None:
            args.prompts_file = "./dataset/ocr/test.txt"
            print("Using default prompts file: ./dataset/ocr/test.txt")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return

        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch = [prompts[i] for i in current_indices]
            process_prompt_batch(batch, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
        print(f"Completed {len(indices)}/{len(indices)} prompts")

    elif args.mode == "eval_partiprompt":
        if args.prompts_file is None:
            args.prompts_file = "./dataset/PartiPrompts.tsv"
            print("Using default prompts file: ./dataset/PartiPrompts.tsv")

        print(f"Loading local TSV: {args.prompts_file}")
        prompts = []
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if header is None:
                raise ValueError("TSV file appears to be empty or missing header")
            prompt_idx = None
            for idx, name in enumerate(header):
                if str(name).strip().lower() == 'prompt':
                    prompt_idx = idx
                    break
            if prompt_idx is None:
                raise ValueError(f"Could not find 'Prompt' column in TSV header: {header}")
            for row in reader:
                if not row or len(row) <= prompt_idx:
                    continue
                text = str(row[prompt_idx]).strip()
                if text:
                    prompts.append(text)
        print(f"Loaded {len(prompts)} PartiPrompts from column 'Prompt'")

        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch_texts = [prompts[i] for i in current_indices]
            process_prompt_batch(batch_texts, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")

    else:
        prompt = args.prompt if args.prompt else "Side-profile panning shot of an anthropomorphic tiger driving a high-speed Formula 1 car, sleek aerodynamic body, 'MapReduce LoRA' clearly printed on the side livery, major F1 circuit, tire smoke, strong motion blur in the background, sharp focus on the car, cinematic lighting, realistic motorsport photography."
        #"Astronaut in a sleek, futuristic spacesuit standing on the rugged Martian surface, with a clear 'MapReduce LoRA' patch emblazoned on the chest, overlooking a vast, dusty red landscape stretching to the horizon beneath a pale, desaturated sky."

        print(f"Using default prompt: {prompt}")
        process_prompt_item(prompt, 0, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
        print("Single prompt processing completed")


if __name__ == "__main__":
    main()


