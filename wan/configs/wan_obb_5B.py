# OBB-driven video: Wan2.2 TI2V-5B backbone + OBB conditioning task.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

# ------------------------ Wan OBB 5B (TI2V-5B backbone) ------------------------#

obb_5B = EasyDict(__name__='Config: Wan OBB 5B')
obb_5B.update(wan_shared_cfg)

# t5
obb_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
obb_5B.t5_tokenizer = 'google/umt5-xxl'

# vae
obb_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
obb_5B.vae_stride = (4, 16, 16)

# transformer (same as TI2V-5B)
obb_5B.patch_size = (1, 2, 2)
obb_5B.dim = 3072
obb_5B.ffn_dim = 14336
obb_5B.freq_dim = 256
obb_5B.num_heads = 24
obb_5B.num_layers = 30
obb_5B.window_size = (-1, -1)
obb_5B.qk_norm = True
obb_5B.cross_attn_norm = True
obb_5B.eps = 1e-6

# OBB conditioning (consumed by WanModel_OBB)
obb_5B.obb_inject_layers = list(range(0, 30, 2))   # every other layer (15 injectors)
obb_5B.obb_d_app = 32                               # per-object appearance latent dim
obb_5B.obb_id_freqs = 64                            # random-Fourier OBB id (table-free, no cap)
obb_5B.obb_depth_bands = 8                          # Mellin bands for entry depth
obb_5B.obb_cond_scale = 8                           # rasterize OBB footprint at cond_scale x latent grid

# inference
obb_5B.sample_fps = 24
obb_5B.sample_shift = 5.0
obb_5B.sample_steps = 50
obb_5B.sample_guide_scale = 5.0
obb_5B.frame_num = 121
