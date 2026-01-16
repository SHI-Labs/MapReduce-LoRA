import ml_collections
import importlib.util
import os

spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    config.train.all_control_tokens = ["<alt>", "<picks>", "<ocr>"]
    config.use_lora = False
    config.train.ema=False
    config.train.train_embeddings = True

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std=True

    config.train.control_token = "<alt>"
    config.teacher_lora_dir = "/logs/geneval_1.0/sd35M-grpo-32gpus/beta0.04_lr0.0003_bz3x32gpus/wandbID-qg5aqeey/checkpoint-4200/lora"
    config.teacher_num_steps = config.sample.num_steps
    config.teacher_guidance_scale = config.sample.guidance_scale

    config.save_freq = 20 # epoch
    config.eval_freq = 20
    # config.save_dir = 'logs/geneval/sd3.5-M-sft-ti'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    config.resolution = 512
    config.sample.train_batch_size = 24
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    
    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000
    
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0   
    config.sample.global_std=True

    config.train.control_token = "<picks>"
    config.teacher_lora_dir = "/logs/pickscore_1.0/sd35M-grpo-32gpus/beta0.01_lr0.0003_bz4x32gpus/wandbID-5p8x9ql5/checkpoint-4500/lora"
    config.teacher_num_steps = config.sample.num_steps
    config.teacher_guidance_scale = config.sample.guidance_scale

    config.save_freq = 20 # epoch
    config.eval_freq = 20
    # config.save_dir = 'logs/pickscore/sd3.5-M-sft'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 24
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    
    config.train.control_token = "<ocr>"
    config.teacher_lora_dir = "/logs/ocr_1.0/sd35M-grpo-32gpus/beta0.04_lr0.0003_bz9x32gpus/wandbID-a4ltbyqr/checkpoint-1600/lora"
    config.teacher_num_steps = config.sample.num_steps
    config.teacher_guidance_scale = config.sample.guidance_scale

    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    # config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_flux():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    # config.train.learning_rate = 1e-5

    config.train.control_token = "<alt>"
    config.teacher_lora_dir = "/logs/geneval_1.0/flux.1-dev-grpo-32gpus/beta0.04_lr0.0003_bz3x32gpus/wandbID-w8p95hlk/checkpoint-2700/lora"
    config.teacher_num_steps = config.sample.num_steps
    config.teacher_guidance_scale = config.sample.guidance_scale

    config.mixed_precision = "bf16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    # config.save_dir = 'logs/geneval/sd3.5-M-sft-ti'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_flux():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True

    config.train.control_token = "<picks>"
    config.teacher_lora_dir = "/logs/pickscore_1.0/flux.1-dev-grpo-32gpus/beta0.0_lr0.0003_bz3x32gpus/wandbID-aeyjjss8/checkpoint-1250/lora"
    config.teacher_num_steps = config.sample.num_steps
    config.teacher_guidance_scale = config.sample.guidance_scale

    config.mixed_precision = "bf16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    # config.save_dir = 'logs/pickscore/flux-group24'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_flux():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    
    config.train.control_token = "<ocr>"
    config.teacher_lora_dir = "/logs/ocr_1.0/flux.1-dev-grpo-32gpus/beta0.04_lr0.0003_bz3x32gpus/wandbID-f4z754z0/checkpoint-1250/lora"
    config.teacher_num_steps = config.sample.num_steps
    config.teacher_guidance_scale = config.sample.guidance_scale

    config.mixed_precision = "bf16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    # config.save_dir = 'logs/pickscore/flux-group24'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()