import os
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from peft import PeftModel
import argparse

def merge_three_loras_to_single(lora_paths, base_model, output_dir, weights=None):
    """
    Merge exactly three LoRA checkpoints by weighted-averaging their adapter weights
    and save the result as a single LoRA checkpoint directory for future loading.

    Args:
        lora_paths: List of exactly three paths to LoRA checkpoint directories.
        base_model: Base transformer model to which LoRA adapters apply.
        output_dir: Destination directory to save the merged LoRA checkpoint.
        weights: Optional sequence of three non-negative numbers representing
            the relative weights for the three LoRA adapters. They will be
            normalized to sum to 1. If None, defaults to equal weights.

    Returns:
        The output_dir where the merged LoRA is saved.
    """
    if not isinstance(lora_paths, (list, tuple)) or len(lora_paths) != 3:
        raise ValueError("merge_three_loras_to_single requires exactly three LoRA paths")

    missing_paths = [path for path in lora_paths if not os.path.exists(path)]
    if missing_paths:
        raise ValueError(f"LoRA checkpoint paths not found: {missing_paths}")

    # Validate and normalize weights
    if weights is None:
        weights = [1.0, 1.0, 1.0]
    if not isinstance(weights, (list, tuple)) or len(weights) != 3:
        raise ValueError("weights must be a sequence of 3 numbers")
    try:
        weights = torch.tensor([float(w) for w in weights], dtype=torch.float32)
    except Exception:
        raise ValueError("weights must be numeric")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total_weight = torch.sum(weights)
    if total_weight.item() == 0:
        raise ValueError("sum of weights must be > 0")
    weights = weights / total_weight
    print(f"Using normalized weights: {weights.tolist()}")

    print(f"Merging 3 LoRA weights into a single adapter at: {output_dir}")

    # Load first LoRA as the working model
    print(f"Loading base LoRA from: {lora_paths[0]}")
    model = PeftModel.from_pretrained(base_model, lora_paths[0])
    base_state_dict = model.state_dict()

    # Load the other two LoRA state dicts
    all_state_dicts = [base_state_dict]
    for i, lora_path in enumerate(lora_paths[1:], 2):
        print(f"Loading LoRA {i}/3 from: {lora_path}")
        temp_model = PeftModel.from_pretrained(base_model, lora_path)
        all_state_dicts.append(temp_model.state_dict())
        del temp_model

    # Compute weighted-averaged adapter weights
    print("Computing weighted adapter weights (3-way)...")
    averaged_state_dict = {}
    keys = base_state_dict.keys()
    for key in keys:
        if 'lora' in key.lower():
            # print("---- key = {} ---".format(key))
            stacked_tensors = torch.stack([sd[key] for sd in all_state_dicts])
            # print("stacked_tensors.size() = ", stacked_tensors.size())
            # Broadcast weights across parameter dimensions and sum
            weight_view = weights.view(-1, *([1] * (stacked_tensors.dim() - 1))).to(stacked_tensors.device)
            # print("weight_view.size() = ", weight_view.size())
            # print("weight_view = ", weight_view)
            averaged_state_dict[key] = torch.sum(stacked_tensors * weight_view, dim=0)
            # print("averaged_state_dict[key].size() = ", averaged_state_dict[key].size())
        else:
            averaged_state_dict[key] = base_state_dict[key]

    # Load averaged weights into the working model (adapter params) and save
    model.load_state_dict(averaged_state_dict, strict=False)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    print(f"Merged LoRA saved to: {output_dir}")
    return output_dir
 
 
def main():
    parser = argparse.ArgumentParser(description='Merge three LoRA checkpoints into a single adapter')
    parser.add_argument('--model_name', type=str, default="sd35m", help='flux.1-dev or sd35m')
    parser.add_argument('--model_path', type=str, default="", help='Optional local snapshot path for base model')
    parser.add_argument('--lora_paths', nargs=3, required=True, help='Three LoRA directories to merge')
    parser.add_argument('--weights', nargs=3, type=float, default=[1.0, 1.0, 1.0], help='Optional weights (3 numbers)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save merged LoRA')

    args = parser.parse_args()

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

    # Load a minimal base to host adapters (CPU only)
    if args.model_name == "flux.1-dev":
        try:
            if args.model_path:
                print("Loading FLUX base from provided path on CPU...")
                pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
            else:
                print("Attempting to load FLUX from local path...")
                pipe = FluxPipeline.from_pretrained(get_hf_hub_cache_dir() / "models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21", torch_dtype=torch.float32)
                print("Successfully loaded from local path")
        except Exception as e:
            print(f"Failed to load from local path: {e}")
            print("Falling back to HuggingFace Hub...")
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32)
            print("Successfully loaded from HuggingFace Hub")
    elif args.model_name == "sd35m":
        try:
            if args.model_path:
                print("Loading SD3.5M base from provided path on CPU...")
                pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
            else:
                print("Attempting to load SD3.5M from local path...")
                pipe = StableDiffusion3Pipeline.from_pretrained(get_hf_hub_cache_dir() / "models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80", torch_dtype=torch.float32)
                print("Successfully loaded from local path")
        except Exception as e:
            print(f"Failed to load from local path: {e}")
            print("Falling back to HuggingFace Hub...")
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float32)
            print("Successfully loaded from HuggingFace Hub")
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    # Safety checker not needed for weight merging
    pipe.safety_checker = None

    merge_three_loras_to_single(
        args.lora_paths,
        pipe.transformer,   # base model
        args.output_dir,
        weights=args.weights,
    )

if __name__ == "__main__":
    main()


