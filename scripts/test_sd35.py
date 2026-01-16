import torch
from diffusers import StableDiffusion3Pipeline
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
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob

# global seed to mirror train eval seeding before any resize_token_embeddings
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def ensure_control_token_in_prompts(prompts, token):
    """Legacy function for single token - kept for backward compatibility"""
    if not token:
        return prompts
    return [p if token in p else (p + " " + token) for p in prompts]

def parse_token_spec(spec: str):
    """
    Parse a combined order+count specification.

    Examples:
      - "<ocr> <picks> <alt>" -> order [<ocr>, <picks>, <alt>], counts default 1 each
      - "<picks> 3 <ocr> 2 <alt> 2" -> order [<picks>, <ocr>, <alt>], counts accordingly
      - Mixed forms like "<alt>:2 <ocr> 1 <picks>=4" are supported

    Returns (order_list, counts_dict)
    """
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
            # token possibly followed by a number
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
    # Default any missing counts to 1
    for t in order:
        counts.setdefault(t, 1)
    return order, counts

def parse_token_weights(spec: str):
    """
    Parse per-token floating-point weights.

    Accepted forms:
      - "<alt>:1.5 <ocr>:0.8"
      - "<alt>=1.5 <ocr>=0.8"
      - "<alt> 1.5 <ocr> 0.8"

    Returns a dict like {"<alt>": 1.5, "<ocr>": 0.8}
    """
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
            # Pair form: "<alt> 1.5"
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
    """
    Add control tokens to prompts based on strategy
    
    Args:
        prompts: List of prompt strings
        available_tokens: List of available control tokens (only loaded ones)
        strategy: "all" (add all loaded tokens), "first" (add first token), "smart" (context-based)
        token_order: Optional list specifying the order of tokens in prompt
        token_counts: Optional dict mapping token -> repetition count (>=1)
    """
    if not available_tokens:
        return prompts
    
    token_counts = token_counts or {}

    result = []
    for prompt in prompts:
        # Check if any control token is already in the prompt
        existing_tokens = [token for token in available_tokens if token in prompt]
        
        # Remove existing tokens first (if any)
        temp_prompt = prompt
        for token in existing_tokens:
            temp_prompt = temp_prompt.replace(token, "").strip()
        # Clean up extra spaces
        temp_prompt = " ".join(temp_prompt.split())
        
        # Determine token order
        if token_order:
            # Use provided order, only include tokens that are actually available
            # Do NOT append remaining available tokens; token_spec should be authoritative
            final_tokens = [token for token in token_order if token in available_tokens]
        else:
            # If no explicit order provided, use all available tokens in their load order
            final_tokens = available_tokens
        
        # Apply repetition counts
        expanded_tokens = []
        for t in final_tokens:
            n = int(max(1, token_counts.get(t, 1)))
            expanded_tokens.extend([t] * n)

        # Add tokens based on strategy
        if strategy == "all":
            # Add all loaded tokens in specified order with repetition
            prompt = temp_prompt + " " + " ".join(expanded_tokens)
        elif strategy == "first":
            # Add first loaded token (respecting order)
            first_t = expanded_tokens[0] if expanded_tokens else final_tokens[0]
            prompt = temp_prompt + " " + first_t
        elif strategy == "smart":
            # Add smart choice based on content
            chosen_token = choose_token_by_content(temp_prompt, final_tokens)
            n = int(max(1, token_counts.get(chosen_token, 1)))
            prompt = temp_prompt + " " + " ".join([chosen_token] * n)
        
        result.append(prompt)
    
    return result

def choose_token_by_content(prompt, available_tokens):
    """Choose the most appropriate control token based on prompt content"""
    prompt_lower = prompt.lower()
    
    # Simple heuristics - you can make this more sophisticated
    if any(word in prompt_lower for word in ['text', 'word', 'letter', 'sign', 'read']):
        if '<ocr>' in available_tokens:
            return '<ocr>'
    
    if any(word in prompt_lower for word in ['quality', 'beautiful', 'aesthetic', 'good', 'best']):
        if '<picks>' in available_tokens:
            return '<picks>'
    
    # Default to first available token
    return available_tokens[0]

class MultiEmbeddingManager:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers
        self.control_tokens = {}  # token -> {token_ids: [], params: [], hooks: []}
        self.enabled = True
        self.token_scales = {}  # token -> float scale

    def add_control_token(self, control_token, embedding_path=None, encoders=None):
        """Add a control token and optionally load its embedding"""
        if control_token in self.control_tokens:
            print(f"Control token {control_token} already exists, replacing...")
            self._remove_hooks(control_token)
        
        # Initialize storage for this token
        self.control_tokens[control_token] = {
            'token_ids': [],
            'params': [],
            'hooks': []
        }
        
        # Ensure token exists in all tokenizers
        self._ensure_token_exists(control_token, encoders)
        
        # Load embedding from file if provided
        if embedding_path and os.path.exists(embedding_path):
            state_dict = torch.load(embedding_path, map_location="cpu")
            self._load_token_embedding(control_token, state_dict)
            print(f"Loaded embedding for {control_token} from {embedding_path}")
        elif embedding_path:
            print(f"Warning: No embedding file found for {control_token} at {embedding_path}")

    def _ensure_token_exists(self, control_token, encoders=None):
        """Ensure control token exists in all tokenizers and encoders"""
        token_data = self.control_tokens[control_token]
        token_data['token_ids'] = []
        token_data['params'] = []
        
        for idx, tok in enumerate(self.tokenizers):
            # Add token if missing
            if control_token not in tok.get_vocab():
                tok.add_tokens([control_token])
                if encoders and idx < len(encoders):
                    try:
                        encoders[idx].resize_token_embeddings(len(tok))
                    except Exception:
                        pass
            
            tid = tok.convert_tokens_to_ids(control_token)
            token_data['token_ids'].append(tid)
            
            # Create parameter - use existing embedding as initialization if encoders provided
            if encoders and idx < len(encoders):
                emb = encoders[idx].get_input_embeddings()
                init_vec = emb.weight[tid].detach().clone().to(dtype=torch.float32)
            else:
                # Fallback to random initialization with typical SD3.5 embedding dimension
                init_vec = torch.randn(4096, dtype=torch.float32) * 0.02
            
            token_data['params'].append(torch.nn.Parameter(init_vec))

    def _load_token_embedding(self, control_token, state_dict):
        """Load embedding for a specific control token"""
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
        """Attach hooks for all control tokens"""
        # Clear all existing hooks
        for control_token in self.control_tokens:
            self._remove_hooks(control_token)
        
        # Resize embeddings if needed
        for idx, (tok, enc) in enumerate(zip(self.tokenizers, encoders)):
            try:
                enc.resize_token_embeddings(len(tok))
            except Exception:
                pass
        
        # Attach hooks for each control token
        for control_token, token_data in self.control_tokens.items():
            self._attach_token_hooks(control_token, encoders)

    def _attach_token_hooks(self, control_token, encoders):
        """Attach hooks for a specific control token"""
        token_data = self.control_tokens[control_token]
        
        for tid, enc, param in zip(token_data['token_ids'], encoders, token_data['params']):
            emb = enc.get_input_embeddings()
            
            # Move parameter to correct device
            if param.device != emb.weight.device:
                with torch.no_grad():
                    param.data = param.data.to(device=emb.weight.device)

            def _hook(module, inputs, output, tid=tid, param=param, token_name=control_token, self_ref=self):
                if not self.enabled:
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
        """Remove hooks for a specific control token"""
        if control_token in self.control_tokens:
            for hook in self.control_tokens[control_token]['hooks']:
                try:
                    hook.remove()
                except Exception:
                    pass
            self.control_tokens[control_token]['hooks'] = []

    def get_token_ids(self, control_token):
        """Get token IDs for a specific control token"""
        if control_token in self.control_tokens:
            return self.control_tokens[control_token]['token_ids']
        return []

    def list_tokens(self):
        """List all loaded control tokens"""
        return list(self.control_tokens.keys())

    def load_state_dict(self, state_dict, control_token, strict: bool = False):
        """Load state dict for a specific control token (backward compatibility)"""
        if control_token in self.control_tokens:
            self._load_token_embedding(control_token, state_dict)
        else:
            print(f"Warning: Control token {control_token} not found in manager")

# Keep the old EmbeddingManager for backward compatibility
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
            # add token if missing and resize embeddings
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
        # clear existing hooks
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
                if not self.enabled:
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
    """
    Load multiple LoRA checkpoints and average their weights
    
    Args:
        lora_paths: List of paths to LoRA checkpoint directories
        base_model: Base transformer model to apply LoRA to
    
    Returns:
        Model with averaged LoRA weights applied
    """
    if len(lora_paths) == 1:
        # Single LoRA case - use existing logic
        print(f"Loading single LoRA from: {lora_paths[0]}")
        model = PeftModel.from_pretrained(base_model, lora_paths[0])
        return model
    
    print(f"Averaging {len(lora_paths)} LoRA weights...")
    
    # Load first LoRA as base
    print(f"Loading base LoRA from: {lora_paths[0]}")
    model = PeftModel.from_pretrained(base_model, lora_paths[0])
    base_state_dict = model.state_dict()
    
    # Collect all LoRA state dicts
    all_state_dicts = [base_state_dict]
    
    for i, lora_path in enumerate(lora_paths[1:], 1):
        print(f"Loading LoRA {i+1}/{len(lora_paths)} from: {lora_path}")
        temp_model = PeftModel.from_pretrained(base_model, lora_path)
        all_state_dicts.append(temp_model.state_dict())
        del temp_model  # Free memory
    
    # Average the weights
    print("Computing average weights...")
    averaged_state_dict = {}
    for key in base_state_dict.keys():
        if any('lora' in key.lower() for part in key.split('.')):
            # Average LoRA-related parameters
            stacked_tensors = torch.stack([sd[key] for sd in all_state_dicts])
            averaged_state_dict[key] = torch.mean(stacked_tensors, dim=0)
        else:
            # Keep non-LoRA parameters from base
            averaged_state_dict[key] = base_state_dict[key]
    
    # Load averaged weights back into model
    model.load_state_dict(averaged_state_dict)
    print("Successfully averaged LoRA weights")
    
    return model


def load_prompts(prompts_file):
    """Load prompts from different file formats"""
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
                # Load GenAI-Bench JSON format
                prompts_data = json.load(f)
                prompts = []
                
                # Convert dict to list with indices
                for key, entry in prompts_data.items():
                    prompts.append((key, entry))
                
                print(f"Loaded {len(prompts)} prompts from GenAI-Bench format")
                return prompts
            else:
                # Load simple text file format
                prompts = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(prompts)} prompts from text format")
                return prompts

def process_prompt_item(prompt_item, index, pipe_sd35m, results_dir, add_control_token, control_token, num_steps: int, guidance_scale: float, args=None):
    """Process a single prompt item and generate image"""
    
    # Handle GenAI-Bench format vs simple text format
    if isinstance(prompt_item, tuple):
        # GenAI-Bench format: (key, entry_dict)
        key, entry = prompt_item
        prompt = entry['prompt']
        seed = entry["random_seed"]
        print(f"\n--- Processing GenAI-Bench prompt {key}: {prompt} | seed: {seed}")
    else:
        # Simple text format
        prompt = prompt_item
        seed = 42  # default seed
        print(f"\n--- Processing prompt {index:06d}: {prompt} | seed: {seed}")
    
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    # prompt = [p.replace("<alt>", "") for p in prompt]

    if add_control_token:
        # Check if we have multiple loaded tokens (multi mode)
        if hasattr(args, '_loaded_control_tokens') and args._loaded_control_tokens:
            prompt = ensure_control_tokens_with_strategy(
                [prompt],
                args._loaded_control_tokens,
                args.token_strategy,
                getattr(args, '_token_order', None),
                getattr(args, '_token_counts', None)
            )[0]
        else:
            # Single token mode (backward compatibility)
            prompt = ensure_control_token_in_prompts([prompt], control_token)[0]

    print(f"Prompt: {prompt}")

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        [pipe_sd35m.text_encoder, pipe_sd35m.text_encoder_2, pipe_sd35m.text_encoder_3],
        [pipe_sd35m.tokenizer, pipe_sd35m.tokenizer_2, pipe_sd35m.tokenizer_3],
        prompt,
        max_sequence_length=128,
    )
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
        [pipe_sd35m.text_encoder, pipe_sd35m.text_encoder_2, pipe_sd35m.text_encoder_3],
        [pipe_sd35m.tokenizer, pipe_sd35m.tokenizer_2, pipe_sd35m.tokenizer_3],
        [""],
        max_sequence_length=128,
    )
    neg_prompt_embeds = neg_prompt_embeds.repeat(len(prompt_embeds), 1, 1)
    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(len(pooled_prompt_embeds), 1)


    with torch.no_grad():
        images_base, _, _ = pipeline_with_logprob(
            pipe_sd35m,
            prompt_embeds=prompt_embeds.to(pipe_sd35m.device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(pipe_sd35m.device),
            negative_prompt_embeds=neg_prompt_embeds.to(pipe_sd35m.device),
            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds.to(pipe_sd35m.device),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            height=512,
            width=512,
            noise_level=0,
        )
    
    # Save image
    if "geneval" in results_dir:
        image_path = os.path.join(results_dir, f"{index:05d}/samples/{index:05d}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    elif isinstance(prompt_item, tuple):
        # Use GenAI-Bench key as filename
        image_path = os.path.join(results_dir, f"{key}.png")
    else:
        # Use index as filename
        image_path = os.path.join(results_dir, f"{index:06d}.png")
    
    images_base[0].save(image_path)
    print(f"Image saved: {image_path} | prompt: {prompt}")
    
    return {
        "prompt": prompt,
        "seed": seed,
        "image_path": image_path
    }

def process_prompt_batch(prompts_batch, save_indices, pipe_sd35m, results_dir, add_control_token, control_token, num_steps: int, guidance_scale: float, args=None):
    """Process a batch of prompt strings and generate images in one forward pass.

    save_indices: list of global indices to use when saving outputs
    """

    # Ensure control token if needed
    if add_control_token:
        # Check if we have multiple loaded tokens (multi mode)
        if hasattr(args, '_loaded_control_tokens') and args._loaded_control_tokens:
            prompts_batch = ensure_control_tokens_with_strategy(
                prompts_batch,
                args._loaded_control_tokens,
                args.token_strategy,
                getattr(args, '_token_order', None),
                getattr(args, '_token_counts', None)
            )
        else:
            # Single token mode (backward compatibility)
            prompts_batch = ensure_control_token_in_prompts(prompts_batch, control_token)

    # Encode prompts as a batch
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        [pipe_sd35m.text_encoder, pipe_sd35m.text_encoder_2, pipe_sd35m.text_encoder_3],
        [pipe_sd35m.tokenizer, pipe_sd35m.tokenizer_2, pipe_sd35m.tokenizer_3],
        prompts_batch,
        max_sequence_length=128,
    )
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
        [pipe_sd35m.text_encoder, pipe_sd35m.text_encoder_2, pipe_sd35m.text_encoder_3],
        [pipe_sd35m.tokenizer, pipe_sd35m.tokenizer_2, pipe_sd35m.tokenizer_3],
        [""],
        max_sequence_length=128,
    )
    neg_prompt_embeds = neg_prompt_embeds.repeat(len(prompt_embeds), 1, 1)
    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(len(pooled_prompt_embeds), 1)

    # Single generator for the whole batch for efficiency
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1)

    with torch.no_grad():
        images_base, _, _ = pipeline_with_logprob(
            pipe_sd35m,
            prompt_embeds=prompt_embeds.to(pipe_sd35m.device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(pipe_sd35m.device),
            negative_prompt_embeds=neg_prompt_embeds.to(pipe_sd35m.device),
            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds.to(pipe_sd35m.device),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            height=512,
            width=512,
            noise_level=0,
        )

    # Save each image according to the target layout
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
    parser = argparse.ArgumentParser(description='Generate images using SD3.5M with optional LoRA fine-tuning')
    parser.add_argument('--prompts_file', type=str, 
                       default=None,
                       help='Path to the prompts file (default: ../data/GenAI-Bench/genai_image_seed.json)')
    parser.add_argument('--prompt', type=str, 
                       default=None,
                       help='given prompt to generate image')
    parser.add_argument('--results_dir', type=str, default="results/",
                       help='Directory to save results')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                       help='Path to LoRA checkpoint directory (optional)')

    # New arguments for multiple LoRA averaging
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
    
    # Multiple embedding paths for different control tokens
    parser.add_argument('--alt_emb_path', type=str, default="",
                       help='Path to <alt> token embeddings')
    parser.add_argument('--ocr_emb_path', type=str, default="", 
                       help='Path to <ocr> token embeddings')
    parser.add_argument('--picks_emb_path', type=str, default="",
                       help='Path to <picks> token embeddings')
    
    # Mode for embedding loading: 'single' or 'multi'
    parser.add_argument('--embedding_mode', type=str, default='auto', choices=['single', 'multi', 'auto'],
                       help='Embedding loading mode: single (use ti_emb_path + control_token), multi (use specific paths), auto (detect based on provided paths)')
    
    # Strategy for adding control tokens when multiple are available
    parser.add_argument('--token_strategy', type=str, default='all', choices=['first', 'all', 'smart'],
                       help='Strategy for control tokens: all (add all loaded tokens), first (add first loaded token), smart (choose based on content). Note: tokens are added to all prompts when embeddings are loaded')
    
    # token_order is deprecated in favor of token_spec
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Number of prompts to process per forward pass')
    parser.add_argument('--num_shards', type=int, default=1,
                       help='Total number of shards for multi-GPU/process inference')
    parser.add_argument('--shard_idx', type=int, default=0,
                       help='This process shard index [0, num_shards)')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
                       help='Inference dtype')
    parser.add_argument('--enable_xformers', action='store_true',
                       help='Enable xFormers memory efficient attention if available')
    parser.add_argument('--enable_sdpa', action='store_true',
                       help='Prefer PyTorch SDPA kernels when available')
    parser.add_argument('--steps', type=int, default=40,
                       help='num_inference_steps')
    parser.add_argument('--guidance', type=float, default=4.5,
                       help='guidance_scale; set 1.0 to disable CFG')
    # New unified spec and weights
    parser.add_argument('--token_spec', type=str, default=os.environ.get('TOKEN_SPEC', ''),
                       help='Combined order+counts, e.g., "<picks> 3 <ocr> 2 <alt> 2" or "<ocr> <picks> <alt>"')
    parser.add_argument('--token_weights', type=str, default=os.environ.get('TOKEN_WEIGHTS', ''),
                       help='Per-token weights, e.g. "<alt>:1.5 <ocr>:0.8" (scales embeddings)')

    args = parser.parse_args()

    # Create results directory
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

    try:
        print("Attempting to load from local path...")
        pipe = StableDiffusion3Pipeline.from_pretrained(get_hf_hub_cache_dir() / "models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80")
        print("Successfully loaded from local path")
    except Exception as e:
        print(f"Failed to load from local path: {e}")
        print("Falling back to HuggingFace Hub...")
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
        # stabilityai/stable-diffusion-3-medium-diffusers
        print("Successfully loaded from HuggingFace Hub")

    pipe = pipe.to("cuda")
    pipe.safety_checker = None

    if args.use_adapter:
        if args.average_loras and args.lora_checkpoints and len(args.lora_checkpoints) > 1:
            print(f"Loading and averaging {len(args.lora_checkpoints)} LoRA weights for transformer")
            # Validate all paths exist
            missing_paths = [path for path in args.lora_checkpoints if not os.path.exists(path)]
            if missing_paths:
                raise ValueError(f"LoRA checkpoint paths not found: {missing_paths}")
            
            # Load and average multiple LoRAs
            pipe.transformer = average_lora_weights(args.lora_checkpoints, pipe.transformer)
            pipe.transformer.set_adapter("default")
            
        elif args.average_loras and args.lora_checkpoints and len(args.lora_checkpoints) == 1:
            print("Warning: average_loras flag set but only one LoRA path provided. Using single LoRA.")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_checkpoints[0])
            pipe.transformer.set_adapter("default")
            
        elif args.average_loras:
            raise ValueError("average_loras flag requires --lora_checkpoints with multiple paths")
            
        else:
            # Original single LoRA logic
            print("Loading pre-trained lora weights for transformer")
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                args.lora_checkpoint
            )
            pipe.transformer.set_adapter("default")
        
        print("Merging LoRA into base weights for faster inference...")
        pipe.transformer = pipe.transformer.merge_and_unload()
        # Ensure dtype/device match the pipeline for optimal kernels
    pipe.transformer.to(device="cuda", dtype=torch.float32)

    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)
    pipe.text_encoder_2.to(dtype=torch.float32)
    pipe.text_encoder_3.to(dtype=torch.float32)


    if args.add_control_token:
        print("pre-registering all control tokens")
        # Pre-register all control tokens so IDs are fixed
        all_ctrl_tokens = list(dict.fromkeys(getattr(args, "all_control_tokens", [])))
        if all_ctrl_tokens:
            for tok, enc in zip([pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3], [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]):
                to_add = [t for t in all_ctrl_tokens if t not in tok.get_vocab()]
                if to_add:
                    tok.add_tokens(to_add)
                    try:
                        enc.resize_token_embeddings(len(tok))
                    except Exception:
                        pass
            ids = [pipe.tokenizer.convert_tokens_to_ids(t) for t in all_ctrl_tokens]
            print(f"Registered control tokens (tok0): {list(zip(all_ctrl_tokens, ids))}")

        # Determine embedding loading mode
        embedding_paths = {
            "<alt>": args.alt_emb_path,
            "<ocr>": args.ocr_emb_path,
            "<picks>": args.picks_emb_path
        }
        
        # Filter out empty paths
        valid_embedding_paths = {k: v for k, v in embedding_paths.items() if v and os.path.exists(v)}

        print(f"Valid embedding paths: {valid_embedding_paths}")
        
        # Auto-detect mode if not specified
        if args.embedding_mode == 'auto':
            if len(valid_embedding_paths) > 1:
                embedding_mode = 'multi'
            elif args.ti_emb_path and args.control_token:
                embedding_mode = 'single'
            elif len(valid_embedding_paths) == 1:
                embedding_mode = 'multi'  # Use multi even for single token for consistency
            else:
                embedding_mode = 'single'  # Fallback
        else:
            embedding_mode = args.embedding_mode
        
        print(f"Using embedding mode: {embedding_mode}")
        
        if embedding_mode == 'multi':
            # Use MultiEmbeddingManager for multiple or flexible loading
            mgr = MultiEmbeddingManager([pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3])
            
            # Load embeddings for each valid path
            for token, path in valid_embedding_paths.items():
                mgr.add_control_token(token, path, [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3])
            
            # If no specific paths provided but ti_emb_path and control_token are available
            if not valid_embedding_paths and args.ti_emb_path and args.control_token:
                mgr.add_control_token(args.control_token, args.ti_emb_path, [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3])
            
            # Apply per-token weights if provided
            token_weights = parse_token_weights(getattr(args, 'token_weights', ''))
            if token_weights:
                for t, w in token_weights.items():
                    mgr.token_scales[t] = float(w)

            mgr.attach([pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3])
            loaded_tokens = mgr.list_tokens()
            print(f"Loaded control tokens: {loaded_tokens}")
            
            # Store loaded tokens for prompt processing
            args._loaded_control_tokens = loaded_tokens
            # Derive order + counts exclusively from token_spec
            spec_order, spec_counts = parse_token_spec(getattr(args, 'token_spec', ''))
            args._token_order = spec_order if spec_order else None
            args._token_counts = spec_counts
            
        else:
            # Use original EmbeddingManager for single token (backward compatibility)
            if not args.control_token:
                raise ValueError("control_token must be specified for single embedding mode")
            
            mgr = EmbeddingManager([pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3], args.control_token)
            mgr.ensure_token_exists([pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3])
            # Apply weight if provided
            token_weights = parse_token_weights(getattr(args, 'token_weights', ''))
            if token_weights and args.control_token in token_weights:
                mgr.token_scale = float(token_weights[args.control_token])
            
            mgr.attach([pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3])

            print(f"Loading TI embeddings for {args.control_token}")
            if args.ti_emb_path and os.path.exists(args.ti_emb_path):
                mgr.load_state_dict(torch.load(args.ti_emb_path, map_location="cpu"))
                # Store loaded token for prompt processing
                args._loaded_control_tokens = [args.control_token]
            else:
                print(f"Warning: No embedding file found at {args.ti_emb_path}")
                args._loaded_control_tokens = []

    # Optional performance knobs
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
        # Load prompts or use default
        if args.prompts_file == None:
            args.prompts_file = "./geneval/prompts/evaluation_metadata.jsonl"
            print("Using default prompts file: ./geneval/prompts/evaluation_metadata.jsonl")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return
        
        print(f"Processing {len(prompts)} prompts...")
        
        # Shard prompts if requested
        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        # Process in batches while preserving original indices
        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch = [prompts[i] for i in current_indices]
            # Extract text strings (geneval jsonl yields dicts)
            batch_texts = [(item["prompt"] if isinstance(item, dict) and "prompt" in item else item) for item in batch]
            process_prompt_batch(batch_texts, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
            # Save metadata if geneval layout
            if "geneval" in args.results_dir:
                for gi, item in zip(current_indices, batch):
                    meta_path = os.path.join(args.results_dir, f"{gi:05d}", "metadata.jsonl")
                    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                    with open(meta_path, "w") as fp:
                        json.dump(item, fp)
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")
    elif args.mode == "eval_genaibench":
        # Load prompts or use default
        if args.prompts_file == None:
            args.prompts_file = "./data/GenAI-Bench/genai_image_seed.json"
            print("Using default prompts file: ./data/GenAI-Bench/genai_image_seed.json")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return
        
        print(f"Processing {len(prompts)} prompts...")
        
        # Shard prompts if requested
        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch = [prompts[i] for i in current_indices]
            # Extract text strings (GenAI-Bench loader yields tuples (key, entry))
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
        # Load prompts or use default
        if args.prompts_file == None:
            args.prompts_file = "./dataset/pickscore/test.txt"
            print("Using default prompts file: ./dataset/pickscore/test.txt")
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            return
        
        # Shard selection
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
        # Load prompts or use default
        if args.prompts_file == None:
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
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")
    elif args.mode == "eval_partiprompt":
        # Load PartiPrompts from a local TSV file
        if args.prompts_file == None:
            args.prompts_file = "./dataset/PartiPrompts.tsv"
            print("Using default prompts file: ./dataset/PartiPrompts.tsv")

        print(f"Loading local TSV: {args.prompts_file}")

        prompts = []
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if header is None:
                raise ValueError("TSV file appears to be empty or missing header")
            # Find prompt column (case-insensitive match for 'Prompt')
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

        # Shard prompts if requested
        total = len(prompts)
        indices = list(range(total))
        if args.num_shards > 1:
            indices = [i for i in indices if (i % args.num_shards) == args.shard_idx]
        print(f"Processing {len(indices)} prompts...")

        # Process in batches
        bs = max(1, int(getattr(args, 'batch_size', 1)))
        for pos in range(0, len(indices), bs):
            current_indices = indices[pos:pos+bs]
            batch_texts = [prompts[i] for i in current_indices]
            process_prompt_batch(batch_texts, current_indices, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
            print(f"Completed {min(pos+bs, len(indices))}/{len(indices)} prompts")
    else:
        # Use default single prompt for testing
        prompt = args.prompt if args.prompt else "a photo of a suitcase right of a boat"
        print(f"Using default prompt: {prompt}")
        
        # Generate single image
        process_prompt_item(prompt, 0, pipe, args.results_dir, args.add_control_token, args.control_token, num_steps=args.steps, guidance_scale=args.guidance, args=args)
        print("Single prompt processing completed")

if __name__ == "__main__":
    main()