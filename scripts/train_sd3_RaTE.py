import os
import json
import hashlib
import datetime
import time
import contextlib
from collections import defaultdict
from concurrent import futures

from absl import app, flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from ml_collections import config_flags

import torch
import numpy as np
from functools import partial
import tqdm
import tempfile
import random
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
)

# add project root to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
import flow_grpo.rewards
from peft import PeftModel

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{self.k}-num_replicas{num_replicas}-batch-size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device, allow_grad=False):
    # Unwrap potential DDP wrappers: encode_prompt expects modules with .dtype
    unwrapped_encoders = []
    for enc in text_encoders:
        if hasattr(enc, "module"):
            unwrapped_encoders.append(enc.module)
        else:
            unwrapped_encoders.append(enc)
    ctx = torch.enable_grad() if allow_grad else torch.no_grad()
    with ctx:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            unwrapped_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

# ==== Textual-inversion helpers ====

class EmbeddingManager(torch.nn.Module):
    def __init__(self, tokenizers, control_token):
        super().__init__()
        if not control_token:
            raise ValueError("control_token must be provided when train_embeddings is True")
        self.control_token = control_token
        self.tokenizers = tokenizers
        self.token_ids = []
        self.trainable_vectors = torch.nn.ParameterList()
        self._hooks = []
        self.enabled = True

    def ensure_token_exists(self, text_encoders):
        self.token_ids = []
        self.trainable_vectors = torch.nn.ParameterList()
        unwrapped = []
        for enc in text_encoders:
            unwrapped.append(enc.module if hasattr(enc, "module") else enc)
        for idx, (tok, enc) in enumerate(zip(self.tokenizers, unwrapped)):
            tokens = tok.encode(self.control_token, add_special_tokens=False)
            if len(tokens) > 1 or self.control_token not in tok.get_vocab():
                tok.add_tokens([self.control_token])
                try:
                    enc.resize_token_embeddings(len(tok))
                except Exception:
                    pass
            tid = tok.convert_tokens_to_ids(self.control_token)
            if tid in (-1, None) or (hasattr(tok, "unk_token_id") and tid == tok.unk_token_id):
                raise ValueError(f"Failed to get valid token ID for control token '{self.control_token}' in tokenizer {idx}")
            self.token_ids.append(tid)
            emb = enc.get_input_embeddings()
            init_vec = emb.weight[tid].detach().clone().to(dtype=torch.float32)
            print(f"{idx} {init_vec[:5]}")
            self.trainable_vectors.append(torch.nn.Parameter(init_vec))

    def attach(self, text_encoders):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        unwrapped = []
        for enc in text_encoders:
            unwrapped.append(enc.module if hasattr(enc, "module") else enc)
        if not self.token_ids or len(self.token_ids) != len(unwrapped):
            self.ensure_token_exists(unwrapped)
        for idx, (tid, enc, vec) in enumerate(zip(self.token_ids, unwrapped, self.trainable_vectors)):
            emb = enc.get_input_embeddings()
            if vec.device != emb.weight.device:
                with torch.no_grad():
                    vec.data = vec.data.to(device=emb.weight.device)

            def _hook(module, inputs, output, tid=tid, param=vec):
                if not self.enabled:
                    return output
                input_ids = inputs[0]
                if not torch.is_floating_point(output):
                    return output
                mask = (input_ids == tid).unsqueeze(-1)
                if mask.any():
                    expanded = param.view(1, 1, -1).expand(output.shape[0], output.shape[1], output.shape[2])
                    expanded = expanded.to(dtype=output.dtype)
                    mask_full = mask.expand_as(output)
                    new_output = torch.where(mask_full, expanded, output)
                    return new_output
                return output

            self._hooks.append(emb.register_forward_hook(_hook))

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    @contextlib.contextmanager
    def disable(self):
        old = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = old

    @torch.no_grad()
    def state_dict(self, *args, **kwargs):
        return {f"encoder_{i}": p.detach().cpu() for i, p in enumerate(self.trainable_vectors)}

    def load_state_dict(self, state_dict, strict: bool = False):
        for i, p in enumerate(self.trainable_vectors):
            key = f"encoder_{i}"
            if key in state_dict:
                with torch.no_grad():
                    p.copy_(state_dict[key].to(device=p.device, dtype=p.dtype))
        return None


def ensure_control_token_in_prompts(prompts, token):
    if not token:
        return prompts
    return [p if token in p else (p + " " + token) for p in prompts]


def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, control_token):
    # compute text embeddings for negative prompts
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device, allow_grad=False)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    all_rewards_lora = defaultdict(list) if int(global_step) == 0 else None
    images_lora = None
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        # auto-append control token at eval to leverage learned embedding
        prompts = ensure_control_token_in_prompts(prompts, control_token)
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device,
            allow_grad=False,
        )
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        
        # Use fixed seed for deterministic evaluation
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(config.seed)

        with autocast():
            with torch.no_grad():
                with pipeline.transformer.disable_adapter():
                    images, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        generator=generator,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=0,
                    )
                # First eval: also sample with LoRA adapter enabled as control group
                if int(global_step) == 0:
                    pipeline.transformer.set_adapter("learned")
                    gen2 = torch.Generator(device=accelerator.device)
                    gen2.manual_seed(config.seed)
                    images_lora, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        generator=gen2,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=0,
                    )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

        # Compute rewards for LoRA images on first eval
        if images_lora is not None and all_rewards_lora is not None:
            rewards_lora = executor.submit(reward_fn, images_lora, prompts, prompt_metadata, only_strict=False)
            time.sleep(0)
            rewards_lora, _ = rewards_lora.result()
            for key, value in rewards_lora.items():
                rewards_gather_lora = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                all_rewards_lora[key].append(rewards_gather_lora)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    if images_lora is not None:
        last_batch_images_lora_gather = accelerator.gather(torch.as_tensor(images_lora, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
    last_batch_rewards_lora_gather = None
    if images_lora is not None and all_rewards_lora is not None:
        last_batch_rewards_lora_gather = {}
        for key, value in rewards_lora.items():
            last_batch_rewards_lora_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if all_rewards_lora is not None:
        all_rewards_lora = {key: np.concatenate(value) for key, value in all_rewards_lora.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            # Save LoRA control images if present (first eval only)
            if images_lora is not None:
                for idx, index in enumerate(sample_indices):
                    image = last_batch_images_lora_gather[index]
                    pil = Image.fromarray(
                        (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"lora_{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            log_payload = {
                "eval_images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{idx}.jpg"),
                        caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                    )
                    for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                ],
                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
            }
            if images_lora is not None and last_batch_rewards_lora_gather is not None and all_rewards_lora is not None:
                sampled_rewards_lora = [
                    {k: last_batch_rewards_lora_gather[k][index] for k in last_batch_rewards_lora_gather}
                    for index in sample_indices
                ]
                log_payload.update({
                    "lora_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"lora_{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards_lora))
                    ],
                    **{f"eval_reward_lora_{key}": np.mean(value[value != -10]) for key, value in all_rewards_lora.items()},
                })
            wandb.log(log_payload, step=global_step)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, trainable_params, config, optimizer=None, epoch=0, control_token=None, embedding_manager=None):
    save_root = os.path.join(save_dir, f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    if accelerator.is_main_process:
        os.makedirs(save_root, exist_ok=True)
        train_embeddings = getattr(config.train, 'train_embeddings', False)
        if train_embeddings and embedding_manager is not None:
            ti_dir = os.path.join(save_root, "ti")
            os.makedirs(ti_dir, exist_ok=True)
            with open(os.path.join(ti_dir, "control_token.txt"), "w") as f:
                f.write((control_token or "") + "\n")
            emb_rows = {}
            try:
                state = accelerator.unwrap_model(embedding_manager).state_dict()
            except Exception:
                state = embedding_manager.state_dict()
            for k, v in state.items():
                emb_rows[k] = v
            torch.save(emb_rows, os.path.join(ti_dir, "embeddings.pt"))
        else:
            os.makedirs(save_root_lora, exist_ok=True)
            if getattr(config.train, 'ema', False) and ema is not None:
                ema.copy_ema_to(trainable_params, store_temp=True)
            unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
            if getattr(config.train, 'ema', False) and ema is not None:
                ema.copy_temp_to(trainable_params)

        training_state = {
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': None if accelerator.state.deepspeed_plugin else (optimizer.state_dict() if optimizer else None),
            'ema_state_dict': (ema.state_dict() if (ema is not None and getattr(config.train, 'ema', False)) else None),
            'wandb_run_id': wandb.run.id if wandb.run else None,
            'rng_state': torch.get_rng_state(),
            'config_dict': config.to_dict()
        }
        torch.save(training_state, os.path.join(save_root, 'training_state.pt'))
        logger.info(f"Saved checkpoint at step {global_step} with wandb run_id: {wandb.run.id if wandb.run else 'None'}")

def load_training_state(checkpoint_dir, optimizer=None, ema=None, embedding_manager=None):
    training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    if os.path.exists(training_state_path):
        try:
            training_state = torch.load(training_state_path, map_location='cpu')
            if optimizer and training_state.get('optimizer_state_dict'):
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
                logger.info("Loaded optimizer state")
            if ema and training_state.get('ema_state_dict'):
                ema.load_state_dict(training_state['ema_state_dict'])
                logger.info("Loaded EMA state")
            try:
                ti_dir = os.path.join(checkpoint_dir, 'ti')
                emb_path = os.path.join(ti_dir, 'embeddings.pt')
                if os.path.exists(emb_path) and embedding_manager is not None:
                    emb_rows = torch.load(emb_path, map_location='cpu')
                    target = getattr(embedding_manager, "module", embedding_manager)
                    target.load_state_dict(emb_rows)
                    logger.info("Restored textual inversion embeddings for control token (manager)")
            except Exception as e:
                raise RuntimeError(f"Failed to restore textual inversion embeddings: {e}") from e
            if training_state.get('rng_state') is not None:
                torch.set_rng_state(training_state['rng_state'])
                logger.info("Restored RNG state")
            epoch = training_state.get('epoch', 0)
            global_step = training_state.get('global_step', 0)
            logger.info(f"Successfully loaded full checkpoint from step {global_step}, epoch {epoch}")
            return epoch, global_step
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            return 0, 0
    else:
        try:
            ti_dir = os.path.join(checkpoint_dir, 'ti')
            emb_path = os.path.join(ti_dir, 'embeddings.pt')
            if os.path.exists(emb_path) and embedding_manager is not None:
                emb_rows = torch.load(emb_path, map_location='cpu')
                target = getattr(embedding_manager, "module", embedding_manager)
                target.load_state_dict(emb_rows)
                logger.info("Restored textual inversion embeddings only (manager); no training state found")
                return 0, 0
        except Exception as e:
            raise RuntimeError(f"Failed to restore textual inversion embeddings: {e}") from e
        logger.info("No checkpoint data found, starting from scratch")
        return 0, 0

def init_wandb_with_resume(config, wandb_run_id=None, num_gpus=None):
    """initialize wandb, support resume (ported from train_sd3.py)"""
    project_name = f"flow_grpo-sd3M-grpo-{os.path.basename(config.dataset)}-{os.path.basename(config.dataset)}-{num_gpus}gpus"
    try:
        if wandb_run_id:
            wandb.init(
                project=project_name,
                config=config.to_dict(),
                id=wandb_run_id,
                resume="must",
                name=config.run_name,
            )
            logger.info(f"Resumed wandb run: {wandb_run_id}")
            config.save_dir = os.path.dirname(config.resume_from)
            logger.info(f"Final save_dir: {config.save_dir}")
        else:
            wandb.init(
                project=project_name,
                config=config.to_dict(),
                name=config.run_name,
            )
            logger.info(f"Started new wandb run: {wandb.run.id}")
            base_dir = config.save_dir
            if getattr(config, "resume_from", None):
                # If loading from a checkpoint but not resuming wandb, place under the same logs root
                # resume_from looks like .../wandbID-<oldid>/checkpoint-<step>
                # We want the parent of wandbID-<oldid>, i.e., logs root
                base_dir = os.path.dirname(os.path.dirname(config.resume_from))
            config.save_dir = os.path.join(base_dir, f"wandbID-{wandb.run.id}")
            os.makedirs(config.save_dir, exist_ok=True)
            logger.info(f"Final save_dir: {config.save_dir}")
    except Exception as e:
        logger.warning(f"Failed to resume wandb run {wandb_run_id}: {e}")
        try:
            wandb.init(
                project=project_name,
                config=config.to_dict(),
                name=config.run_name,
            )
            logger.info(f"Started new wandb run after resume failed: {wandb.run.id}")
            base_dir = config.save_dir
            if getattr(config, "resume_from", None):
                base_dir = os.path.dirname(os.path.dirname(config.resume_from))
            config.save_dir = os.path.join(base_dir, f"wandbID-{wandb.run.id}")
            os.makedirs(config.save_dir, exist_ok=True)
            logger.info(f"Final save_dir: {config.save_dir}")
        except Exception:
            pass


def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


@torch.no_grad()
def teacher_generate_images(teacher_pipe, prompts, height, width, steps, guidance, gens=None):
    out = teacher_pipe(
        prompts,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gens,
        output_type="pt",
        height=height,
        width=width,
        noise_level=0,
    )
    return out.images


@torch.no_grad()
def encode_to_latents(vae, images):
    imgs = images * 2.0 - 1.0
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.mean
    sf = getattr(vae.config, "scaling_factor", 1.0)
    return latents * sf


def create_generators(prompts, base_seed):
    gens = []
    for p in prompts:
        h = hashlib.sha256(p.encode()).digest()
        s = (base_seed + int.from_bytes(h[:4], "big")) % (2**31)
        gens.append(torch.Generator().manual_seed(s))
    return gens


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    def get_experiment_name(reward_fn):
        """Map reward function to experiment directory name"""
        if not reward_fn:
            return "default"
        
        # track multiple rewards
        # Ex: mapping reward_fn {"geneval": 0.3, "pickscore": 0.7} to "geneval_0.3_pickscore_0.7"
        reward_keys_str = "_".join([f"{key}_{value}" for key, value in reward_fn.items()])
        
        return reward_keys_str
    

    # Dynamic GPU count detection - use actual processes, not total GPUs
    num_gpus = accelerator.num_processes

    # Construct save_dir with actual config values
    experiment_name = get_experiment_name(config.reward_fn)
    config.save_dir = f'./logs/{experiment_name}/sd35M-sft-{num_gpus}gpus/beta{config.train.beta}_lr{config.train.learning_rate}_bz{config.sample.train_batch_size}x{num_gpus}gpus'
    logger.info(f"Auto-constructed save_dir: {config.save_dir}")

    # === Resume / wandb init ===
    resume_checkpoint_dir = None
    wandb_run_id = None
    resume_from = getattr(config, 'resume_from', None)
    if resume_from and os.path.exists(resume_from):
        resume_checkpoint_dir = resume_from
        parts = resume_from.split('/')
        for part in parts:
            if part.startswith('wandbID-'):
                wandb_run_id = part[8:]
                break
        if not wandb_run_id:
            logger.warning(f"Could not extract wandb_run_id from resume_from: {resume_from}")
            logger.warning("Expected path format: .../wandbID-<run_id>/checkpoint-<step>")
            wandb_run_id = None
        # Allow resuming model/optimizer state while starting a fresh wandb run
        if not getattr(config, 'resume_wandb', True):
            logger.info("resume_wandb=False -> will start a NEW wandb run while loading from checkpoint")
            wandb_run_id = None
        logger.info(f"Resuming from: {resume_from}")
    else:
        logger.info("Starting training from scratch")

    if accelerator.is_main_process:
        init_wandb_with_resume(config, wandb_run_id, accelerator.num_processes)

    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    # Student pipeline: base SD3.5M, frozen except token embeddings via hooks
    student = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    student.vae.requires_grad_(False)
    student.text_encoder.requires_grad_(False)
    student.text_encoder_2.requires_grad_(False)
    student.text_encoder_3.requires_grad_(False)
    student.transformer.requires_grad_(False)
    student.safety_checker = None

    student.transformer.train()
    student.transformer.enable_gradient_checkpointing()
    
    # Load teacher LoRA into the same transformer as adapter "learned"
    student.transformer = PeftModel.from_pretrained(
        student.transformer, config.teacher_lora_dir, adapter_name="learned"
    )

    # dtype/device
    inference_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     inference_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     inference_dtype = torch.bfloat16

    student.vae.to(accelerator.device, dtype=torch.float32)
    student.text_encoder.to(accelerator.device, dtype=inference_dtype)
    student.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    student.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    student.transformer.to(accelerator.device)


    # Text encoders / tokenizers
    text_encoders = [student.text_encoder, student.text_encoder_2, student.text_encoder_3]
    tokenizers = [student.tokenizer, student.tokenizer_2, student.tokenizer_3]

    # Pre-register all control tokens so IDs are fixed; only one will be trained
    all_ctrl_tokens = list(dict.fromkeys(getattr(config.train, "all_control_tokens", [])))
    if all_ctrl_tokens:
        for tok, enc in zip(tokenizers, text_encoders):
            to_add = [t for t in all_ctrl_tokens if t not in tok.get_vocab()]
            if to_add:
                tok.add_tokens(to_add)
                enc.resize_token_embeddings(len(tok))
        if accelerator.is_main_process:
            ids = [tokenizers[0].convert_tokens_to_ids(t) for t in all_ctrl_tokens]
            logger.info(f"Registered control tokens (tok0): {list(zip(all_ctrl_tokens, ids))}")

    # Embedding-only training
    control_token = getattr(config.train, "control_token", None)
    if not control_token:
        raise ValueError("config.train.control_token must be set for SFT TI training")
    embedding_manager = EmbeddingManager(tokenizers, control_token)
    embedding_manager.ensure_token_exists(text_encoders)
    trainable_params = list(embedding_manager.parameters())

    # Dataset / DDP
    if config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.train.batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )
        collate_fn = GenevalPromptDataset.collate_fn
    else:
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')
        
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.train.batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )
        collate_fn = TextPromptDataset.collate_fn

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Prepare with accelerator
    embedding_manager, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        embedding_manager, optimizer, train_dataloader, test_dataloader
    )
    accelerator.unwrap_model(embedding_manager).attach(text_encoders)

    autocast = accelerator.autocast
    height = config.resolution
    width = config.resolution

    # Flow-matching scheduler for SD3.5 training objective
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained.model, subfolder="scheduler"
    )

    # Optional async exec for future extension (e.g., caching teacher outputs)
    executor = futures.ThreadPoolExecutor(max_workers=4)

    # Load training state after models/optimizer are ready
    resume_epoch, resume_global_step = 0, 0
    is_first_iteration_after_resume = False
    if resume_checkpoint_dir:
        resume_epoch, resume_global_step = load_training_state(
            resume_checkpoint_dir, optimizer=optimizer, ema=None, embedding_manager=embedding_manager
        )

        logger.info(f"=== Resumed training from checkpoint ===")
        logger.info(f"  Checkpoint: {resume_checkpoint_dir}")
        logger.info(f"  Epoch: {resume_epoch}")
        logger.info(f"  Global step: {resume_global_step}")
        logger.info(f"  Wandb run ID: {wandb.run.id if wandb.run else 'None'}")
        is_first_iteration_after_resume = True
        if not config.resume_wandb:
            is_first_iteration_after_resume = False
    else:
        logger.info("=== Starting training from scratch ===")

    # Precompute negative prompt embeddings for CFG during teacher sampling
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device, allow_grad=False)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    logger.info("***** Running SFT (token-embedding) training *****")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")

    # Reward fns for eval
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    epoch = resume_epoch
    global_step = resume_global_step
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        student.transformer.eval()
        if not is_first_iteration_after_resume:
            if epoch % getattr(config, "eval_freq", 10) == 0:
                eval(student, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, control_token)
            if accelerator.is_main_process and (epoch % getattr(config, "save_freq", 100) == 0) and epoch > 0:
                save_ckpt(
                    config.save_dir,
                    student.transformer,
                    global_step,
                    accelerator,
                    ema=None,
                    trainable_params=trainable_params,
                    config=config,
                    optimizer=optimizer,
                    epoch=epoch,
                    control_token=control_token,
                    embedding_manager=embedding_manager,
                )

        is_first_iteration_after_resume = False
        #################### SAMPLING ####################
        student.transformer.eval()
        collected = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            # Student prompts: auto-append control token for training
            student_prompts = ensure_control_token_in_prompts(prompts, control_token)

            # Teacher latents via pipeline_with_logprob (avoid image decode/encode)
            base_seed = epoch * 10000 + i
            gens = create_generators(prompts, base_seed=base_seed) if getattr(config.sample, "same_latent", False) else None
            with torch.no_grad():
                # ensure teacher embeddings are not affected by control-token replacement hooks
                # explicitly remove control token from teacher prompts
                teacher_prompts = [p.replace(control_token, "").strip() for p in prompts]
                with accelerator.unwrap_model(embedding_manager).disable():
                    teacher_prompt_embeds, teacher_pooled_embeds = compute_text_embeddings(
                        teacher_prompts,
                        text_encoders,
                        tokenizers,
                        max_sequence_length=128,
                        device=accelerator.device,
                        allow_grad=False,
                    )
            neg_embeds_b = train_neg_prompt_embeds[: len(teacher_prompt_embeds)]
            neg_pooled_b = train_neg_pooled_prompt_embeds[: len(teacher_pooled_embeds)]
            with torch.no_grad(), autocast():
                # enable teacher adapter for sampling
                student.transformer.set_adapter("learned")
                _, latents_seq, _ = pipeline_with_logprob(
                    student,
                    prompt_embeds=teacher_prompt_embeds,
                    pooled_prompt_embeds=teacher_pooled_embeds,
                    negative_prompt_embeds=neg_embeds_b,
                    negative_pooled_prompt_embeds=neg_pooled_b,
                    num_inference_steps=config.teacher_num_steps,
                    guidance_scale=config.teacher_guidance_scale,
                    output_type="pt",
                    height=height,
                    width=width,
                    noise_level=0,
                    generator=gens,
                )
            latents = latents_seq[-1]
            collected.append({"latents": latents, "student_prompts": student_prompts})

        # Collate collected samples
        latents_all = torch.cat([c["latents"] for c in collected], dim=0)
        student_prompts_all = sum([c["student_prompts"] for c in collected], [])
        total_batch_size = latents_all.shape[0]

        #################### TRAINING ####################
        # transformer is frozen; keep eval() for stable behavior and base mode
        with student.transformer.disable_adapter():
            for inner_epoch in range(getattr(config.train, "num_inner_epochs", 1)):
                # shuffle
                perm = torch.randperm(total_batch_size, device=accelerator.device)
                latents_all = latents_all[perm]
                idx = perm.cpu().tolist()
                student_prompts_all = [student_prompts_all[j] for j in idx]

                # rebatch by num_batches_per_epoch
                per_batch = total_batch_size // config.sample.num_batches_per_epoch
                latents_batched = latents_all.reshape(-1, per_batch, *latents_all.shape[1:])
                prompt_batches = [student_prompts_all[k:k + per_batch] for k in range(0, total_batch_size, per_batch)]

                info = defaultdict(list)
                for i, (latents_mb, prompts_mb) in enumerate(tqdm(
                    list(zip(latents_batched, prompt_batches)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
                )):
                    # treat teacher latents as fixed targets
                    latents_mb = latents_mb.detach()
                    # Sample timesteps and noise per mini-batch
                    bsz = latents_mb.shape[0]
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme='logit_normal',
                        batch_size=bsz,
                        logit_mean=0,
                        logit_std=1,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    timesteps = noise_scheduler.timesteps[indices].to(device=latents_mb.device)

                    noise = torch.randn_like(latents_mb)
                    sigmas = get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=latents_mb.ndim, dtype=latents_mb.dtype)
                    noisy_model_input = (1.0 - sigmas) * latents_mb + sigmas * noise

                    # Student embeddings with gradients per step
                    prompt_embeds, pooled_embeds = compute_text_embeddings(
                        prompts_mb,
                        text_encoders,
                        tokenizers,
                        max_sequence_length=128,
                        device=accelerator.device,
                        allow_grad=True,
                    )

                    with accelerator.accumulate(embedding_manager):
                        with autocast():
                            model_pred = student.transformer(
                                hidden_states=noisy_model_input,
                                timestep=timesteps,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0]
                        target = noise - latents_mb
                        loss = ((model_pred.float() - target.float()) ** 2).mean()
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(trainable_params, config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            try:
                                wandb.log(info, step=global_step)
                            except Exception:
                                pass
                        info = defaultdict(list)
                        global_step += 1

        epoch += 1


if __name__ == "__main__":
    app.run(main)

