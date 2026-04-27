import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from utils.data_utils import get_data
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as pos_emb_llama
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as pos_emb_qwen
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict
from utils import utils, data_utils, fuse_norm_utils, quant_utils
from tqdm import tqdm
import transformers
from utils.hadamard_utils import random_orthogonal_matrix
import time


def _safe_rotary_emb(rotary_emb, x, position_ids, seqlen):
    """Handle transformers variants that expect position_ids or seq_len."""
    try:
        return rotary_emb(x, position_ids)
    except TypeError:
        return rotary_emb(x, seqlen)
    except RuntimeError as exc:
        # Older Qwen2 rotary embedding compares seq_len as int; Tensor raises this.
        if "Boolean value of Tensor with more than one value is ambiguous" in str(exc):
            return rotary_emb(x, seqlen)
        raise


def quant_rel_error_coeff(bits: int) -> float:
    if bits >= 16:
        return 0.0
    maxq = float((1 << (bits - 1)) - 1)
    return 1.0 / max(maxq * maxq, 1.0)


def fused_input_weight_cov(linear_modules):
    cov = None
    for module in linear_modules:
        weight_t = module.weight.detach().to(dtype=torch.float64).T.cpu()
        cur_cov = weight_t @ weight_t.T
        cov = cur_cov if cov is None else cov + cur_cov
    if cov is None:
        raise RuntimeError("Expected at least one module for fused weight covariance.")
    return cov


def _safe_inverse_scale(scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
    safe_scale = torch.where(scale.abs() < eps, sign * eps, scale)
    return 1.0 / safe_scale


def diag_congruence_transform(
    cov: torch.Tensor, scale: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    s = scale.to(device=cov.device, dtype=torch.float64)
    if inverse:
        s = _safe_inverse_scale(s)
    cov64 = cov.to(torch.float64)
    return (s[:, None] * cov64) * s[None, :]


def collect_pre_fuse_ln_scales(llm: torch.nn.Module):
    scale_map = {}
    if not hasattr(llm, "model") or not hasattr(llm.model, "layers"):
        return scale_map
    for layer_idx, layer in enumerate(llm.model.layers):
        layer_scales = {}
        if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
            layer_scales["self_attn"] = (
                layer.input_layernorm.weight.detach().to(torch.float64).cpu()
            )
        if hasattr(layer, "post_attention_layernorm") and hasattr(
            layer.post_attention_layernorm, "weight"
        ):
            layer_scales["mlp"] = (
                layer.post_attention_layernorm.weight.detach().to(torch.float64).cpu()
            )
        scale_map[layer_idx] = layer_scales
    return scale_map


def _safe_scale_for_mapping(scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
    return torch.where(scale.abs() < eps, sign * eps, scale)


def map_basis_to_fused_orthonormal(
    evec_phys: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    s = _safe_scale_for_mapping(scale.to(device=evec_phys.device, dtype=torch.float64))
    mapped = s[:, None] * evec_phys.to(torch.float64)
    q, r = torch.linalg.qr(mapped, mode="reduced")
    sign = torch.sign(torch.diag(r))
    sign[sign == 0] = 1.0
    q = q * sign.unsqueeze(0)
    return q


def compute_fro_proxy_lambda_terms(
    x_cov: torch.Tensor,
    weight_cov: torch.Tensor,
    low_bits: int,
    high_bits: int,
    high_len: int,
):
    alpha_low_sq = quant_rel_error_coeff(low_bits)
    alpha_high_sq = quant_rel_error_coeff(high_bits)
    beta_low_sq = quant_rel_error_coeff(low_bits)
    beta_high_sq = quant_rel_error_coeff(high_bits)

    d = int(x_cov.shape[0])
    r = int(high_len)
    low_len = max(d - r, 1)
    high_len_safe = max(r, 1)

    gamma_low = (alpha_low_sq + beta_low_sq) / float(low_len)
    gamma_high = (alpha_high_sq + beta_high_sq) / float(high_len_safe)

    x_fro_sq = float(torch.trace(x_cov.to(torch.float64)).item())
    w_fro_sq = float(torch.trace(weight_cov.to(torch.float64)).item())

    lambda_x = gamma_low * w_fro_sq
    lambda_w = gamma_low * x_fro_sq
    return {
        "gamma_low": gamma_low,
        "gamma_high": gamma_high,
        "lambda_x": lambda_x,
        "lambda_w": lambda_w,
    }


def build_o_proj_weight_cov_per_kv_head(
    o_proj: nn.Linear,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Build per-kv-head weight covariance for value path objective.
    In GQA, one kv head fans out to multiple query heads, so we sum the
    corresponding o_proj head-block covariances.
    Returns shape: [num_kv_heads, head_dim, head_dim].
    """
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads({num_attention_heads}) must be divisible by num_kv_heads({num_kv_heads})."
        )
    num_kv_groups = num_attention_heads // num_kv_heads
    w = o_proj.weight.detach().to(torch.float64).cpu()  # [hidden_dim, hidden_dim]
    hidden_in = int(w.shape[1])
    expected_hidden_in = int(num_attention_heads * head_dim)
    if hidden_in != expected_hidden_in:
        raise ValueError(
            f"Unexpected o_proj input dim={hidden_in}, expected={expected_hidden_in} "
            f"(num_attention_heads={num_attention_heads}, head_dim={head_dim})."
        )

    cov = torch.zeros(num_kv_heads, head_dim, head_dim, dtype=torch.float64)
    for kv_h in range(num_kv_heads):
        start_qh = kv_h * num_kv_groups
        end_qh = (kv_h + 1) * num_kv_groups
        for qh in range(start_qh, end_qh):
            col_l = qh * head_dim
            col_r = (qh + 1) * head_dim
            block = w[:, col_l:col_r]  # [hidden_dim, head_dim]
            cov[kv_h] += block.T @ block
    return cov


@torch.no_grad()
def get_basis(model_args, training_args, ptq_args) -> None:
    vision = False
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=torch.float32,
    )
    model.seqlen = training_args.model_max_length
    transformers.set_seed(ptq_args.seed)
    model.eval()

    if hasattr(model, "language_model"): #for vision llama
        llm = model.language_model
    else:
        llm = model

    use_pre_fuse_lambda_compensation = bool(
        getattr(ptq_args, "use_pre_fuse_lambda_compensation", 1)
    )
    use_kv_wa_cov = bool(getattr(ptq_args, "use_kv_wa_cov", 1))
    pre_fuse_ln_scale_map = collect_pre_fuse_ln_scales(llm)

    # Fuse Norm
    fuse_norm_utils.fuse_layer_norms(llm)

    utils.cleanup_memory(verbos=True)

    llm.config.use_cache = False
    seqlen = model.seqlen

    train_data = data_utils.get_data(
        seed=ptq_args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        eval_mode=False,
        nsamples=ptq_args.nsamples,
        calib_dataset=ptq_args.calib_dataset,
        vision=vision,
    )
    nbatches = len(train_data)
    layers = llm.model.layers
    llm.model.embed_tokens = llm.model.embed_tokens.to(utils.DEV)
    if hasattr(llm.model, "rotary_emb"):
        llm.model.rotary_emb = llm.model.rotary_emb.to(utils.DEV)

    layers[0] = layers[0].to(utils.DEV)

    dtype = next(iter(llm.parameters())).dtype

    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, model.seqlen, llm.config.hidden_size),
        dtype=dtype,
        device=utils.DEV,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            if "position_embeddings" in kwargs:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            else:
                cache["position_embeddings"] = None
            cache["cross_attention_states"] = kwargs["cross_attention_states"] if "cross_attention_states" in kwargs.keys() else None
            cache["cross_attention_mask"] = kwargs["cross_attention_mask"] if "cross_attention_mask" in kwargs.keys() else None
            cache["full_text_row_masked_out_mask"] = kwargs["full_text_row_masked_out_mask"] if "full_text_row_masked_out_mask" in kwargs.keys() else None
            cache["cache_position"] = kwargs["cache_position"] if "cache_position" in kwargs.keys() else None
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nbatches):
        batch = train_data[i][0].to(utils.DEV)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    llm.model.embed_tokens = llm.model.embed_tokens.cpu()
    if hasattr(llm.model, "rotary_emb"):
        llm.model.rotary_emb = llm.model.rotary_emb.cpu()
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches

    basis_dict = {}
    eval_dict = {}
    rotation_dict = {}
    attention_mask = cache["attention_mask"]

    hidden_dim = llm.config.hidden_size 
    num_attention_heads = llm.config.num_attention_heads 
    head_dim = hidden_dim // num_attention_heads 
    kv_heads = llm.config.num_key_value_heads 
    down_proj_blocksize = ptq_args.down_proj_blocksize
    nlayers = len(layers)
    
    low_frac, high_frac = ptq_args.low_fraction, ptq_args.high_fraction
    low_length_hidden, high_length_hidden = int(low_frac * hidden_dim), int(high_frac * hidden_dim) 
    low_length_head, high_length_head = int(low_frac * head_dim), int(high_frac * head_dim) 
    low_length_down_proj, high_length_down_proj = int(low_frac * down_proj_blocksize), int(high_frac * down_proj_blocksize)

    rotation_granularity = ptq_args.rotation_granularity

    sparse_fraction = ptq_args.sparse_fraction
    sparse_length_hidden = int(sparse_fraction * low_length_hidden)
    sparse_length_head = int(sparse_fraction * low_length_head)

    if '70b' in model_args.input_model.lower() or "72b" in model_args.input_model.lower() :
        cov_device = 'cpu'
    else:
        cov_device = utils.DEV

    # initialize covariance matrices
    H_attn = torch.zeros((len(layers), hidden_dim, hidden_dim), device=cov_device)
    H_mlp = torch.zeros((len(layers), hidden_dim, hidden_dim), device=cov_device)
    H_down_proj = torch.zeros(
        (
            nlayers,
            down_proj_blocksize,
            down_proj_blocksize,
        ),
        device=cov_device,
    )  # block down_proj with certain dimension
    H_value = torch.zeros(
        (
            nlayers,
            llm.config.num_key_value_heads,
            head_dim,
            head_dim,
        ),
        device=cov_device,
    )
    H_key_pos = torch.zeros(
        (
            nlayers,
            kv_heads,
            head_dim,
            head_dim,
        ),
        device=cov_device,
    )
    H_query_pos = torch.zeros(
        (
            nlayers,
            kv_heads,
            head_dim,
            head_dim,
        ),
        device=cov_device,
    )
    R1_1 = random_orthogonal_matrix(
        hidden_dim - high_length_hidden - low_length_hidden - sparse_length_hidden, "cuda"
    )

    rotation_dict["R1_1"] = R1_1
    rotation_dict["R1_2"] = random_orthogonal_matrix(high_length_hidden, "cuda")
    if low_length_hidden != 0 :
        R1_0 = random_orthogonal_matrix(low_length_hidden - sparse_length_hidden, "cuda")
        if sparse_length_hidden > 0:
            zeros = torch.zeros(
                (sparse_length_hidden, sparse_length_hidden),
                device=R1_1.device,
                dtype=R1_1.dtype,
            )
            R1_0 = torch.block_diag(zeros, R1_0)
        rotation_dict["R1_0"] = R1_0
    else:
        rotation_dict["R1_0"] = None


    R2_1 = random_orthogonal_matrix(
        head_dim - high_length_head - low_length_head,
        "cuda",
    )
    rotation_dict["R2_1"] = R2_1
    rotation_dict["R2_2"] = random_orthogonal_matrix(high_length_head, "cuda")
    if low_length_head != 0 :
        R2_0 = random_orthogonal_matrix(low_length_head - sparse_length_head, "cuda")
        if sparse_length_hidden > 0:
            zeros = torch.zeros(
                (sparse_length_head, sparse_length_head),
                device=R2_1.device,
                dtype=R2_1.dtype,
            )
            R2_0 = torch.block_diag(zeros, R2_0)
        rotation_dict["R2_0"] = R2_0
    else:
        rotation_dict["R2_0"] = None


    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    rotation_path = os.path.join(
        model_args.output_rotation_path,
        "R-high-"
        + str(high_frac)
        + "-low-"
        +str(low_frac)
        + "-sparse-"
        + str(sparse_fraction)
        + "-"
        + model_args.input_model.split("/")[-2]
        + "-"
        + model_args.input_model.split("/")[-1]
        + ".bin",
    )
    if not os.path.exists(rotation_path):
        torch.save(
            rotation_dict,
            rotation_path,
        )
    basis_path = os.path.join(
        model_args.output_rotation_path,
        "U-"
        + str(ptq_args.basis_cov_mode)
        + "-"
        + str(ptq_args.calib_dataset)
        + "-"
        + str(ptq_args.nsamples)
        + "-"
        + model_args.input_model.split("/")[-2]
        + "-"
        + model_args.input_model.split("/")[-1]
        + ".bin",
    )

    eval_path = os.path.join(
        model_args.output_rotation_path,
        "E-"
        + str(ptq_args.basis_cov_mode)
        + "-"
        + str(ptq_args.calib_dataset)
        + "-"
        + str(ptq_args.nsamples)
        + "-"
        + model_args.input_model.split("/")[-2]
        + "-"
        + model_args.input_model.split("/")[-1]
        + ".bin",
    )
    if not os.path.exists(basis_path):
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        for i in tqdm(range(nlayers), desc="(Collecting Cov matrices) Layers"):
            layer = layers[i].to(utils.DEV)

            hooks = []
            #捕获1.QKV各自的output和共同的input 2.up和down的input
            def hook_fn_upproj(module, input, output):
                global input_up_proj
                input_up_proj = input[0]

            def hook_fn_vproj(module, input, output):
                global output_vproj
                output_vproj = output

            def hook_fn_kproj(module, input, output):
                global output_kproj
                output_kproj = output

            def hook_fn_qproj(module, input, output):
                global output_qproj, input_qkv_proj
                output_qproj = output
                input_qkv_proj = input[0]

            def hook_fn_downproj(module, input, output):
                global input_down_proj
                input_down_proj = input[0]

            hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn_upproj))
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn_downproj))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn_kproj))
            hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj))

            for j in range(nbatches):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
                    
                    # reshape to get value states per head
                    value_states = output_vproj.view(
                        1,
                        seqlen,
                        kv_heads,
                        head_dim,
                    ).transpose(1, 2) #变为[1,kv_heads,seqlen,head_dim]的张量
                    # rope cos, sin
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                    else:
                        cos, sin = _safe_rotary_emb(
                            layer.self_attn.rotary_emb,
                            value_states,
                            position_ids,
                            seqlen,
                        )
                    # reshape to get key states per head
                    key_states = output_kproj.view(
                        1,
                        seqlen,
                        kv_heads,
                        head_dim,
                    ).transpose(1, 2) #变为[1,kv_heads,seqlen,head_dim]的张量
                    # key_states = repeat_kv(key_states, layer.self_attn.num_key_value_groups)

                    # reshape to get query states per head
                    query_states = output_qproj.view(
                        1,
                        seqlen,
                        num_attention_heads,
                        head_dim,
                    ).transpose(1, 2) #变为[1,num_attention_heads,seqlen,head_dim]的张量

                    if "llama" in model_args.input_model.lower() and "vision" not in model_args.input_model.lower():
                        query_states_pos, key_states_pos = pos_emb_llama(
                            query_states, key_states, cos, sin
                        )
                    elif "qwen" in model_args.input_model.lower() and "vl" not in model_args.input_model.lower():
                        query_states_pos, key_states_pos = pos_emb_qwen(
                            query_states, key_states, cos, sin, position_ids
                        )

                    # calculate covariance
                    H_mlp[i] += torch.sum(
                        input_up_proj.double().mT @ input_up_proj.double(), dim=0
                    ).to(cov_device)  # shape : [hidden_dim, hidden_dim] 捕获的是mlp模块的输入，input_up_proj是1*2048*2048的

                    H_attn[i] += torch.sum(
                        input_qkv_proj.double().mT @ input_qkv_proj.double(), dim=0
                    ).to(cov_device)  # shape : [hidden_dim, hidden_dim]
                    H_value[i] += torch.sum(
                        value_states.double().mT @ value_states.double(), dim=0 #value_states是[1,kv_heads,seqlen,head_dim]的张量，捕获value的输出
                    ).to(cov_device)  # shape : [num_heads, head_dim, head_dim]

                    H_key_pos[i] += torch.sum(
                        key_states_pos.double().mT @ key_states_pos.double(), dim=(0) #key_states_pos是[1,kv_heads,seqlen,head_dim]的张量，经过RoPE后的输出
                    ).to(cov_device)  # shape : [num_kv_heads, head_dim, head_dim]

                    # For K objective in kv wa_cov mode: aggregate query covariance to kv-head granularity.
                    query_states_pos_grouped = query_states_pos.contiguous().view(
                        1,
                        kv_heads,
                        num_attention_heads // kv_heads,
                        seqlen,
                        head_dim,
                    ).reshape(
                        1,
                        kv_heads,
                        (num_attention_heads // kv_heads) * seqlen,
                        head_dim,
                    )
                    H_query_pos[i] += torch.sum(
                        query_states_pos_grouped.double().mT @ query_states_pos_grouped.double(),
                        dim=0,
                    ).to(cov_device)  # shape : [num_kv_heads, head_dim, head_dim]

                    H_down_proj[i] += torch.sum(
                        input_down_proj.view(
                            input_down_proj.shape[0], -1, ptq_args.down_proj_blocksize
                        )
                        .double()
                        .mT
                        @ input_down_proj.view(
                            input_down_proj.shape[0], -1, ptq_args.down_proj_blocksize
                        ).double(),
                        dim=(0),
                    ).to(cov_device)  # shape : [1024, 1024]
            for hook in hooks:
                hook.remove()

            layers[i] = layers[i].cpu()

            torch.cuda.empty_cache()

            inps, outs = outs, inps

        for i in tqdm(range(nlayers), desc="(Getting Basis) Layers"):

            # eigen decomposition of attn
            attn_cov = H_attn[i] / (nbatches * seqlen)
            if ptq_args.basis_cov_mode == "wa_cov":
                attn_weight_cov = fused_input_weight_cov(
                    [
                        layers[i].self_attn.q_proj,
                        layers[i].self_attn.k_proj,
                        layers[i].self_attn.v_proj,
                    ]
                )
                attn_ln_scale = pre_fuse_ln_scale_map.get(i, {}).get("self_attn")
                if attn_ln_scale is None or int(attn_ln_scale.numel()) != int(attn_cov.shape[0]):
                    attn_ln_scale = torch.ones(int(attn_cov.shape[0]), dtype=torch.float64)
                if use_pre_fuse_lambda_compensation:
                    attn_cov_for_lambda = diag_congruence_transform(
                        attn_cov.cpu(), attn_ln_scale, inverse=False
                    )
                    attn_weight_cov_for_lambda = diag_congruence_transform(
                        attn_weight_cov, attn_ln_scale, inverse=True
                    )
                else:
                    attn_cov_for_lambda = attn_cov.cpu().to(torch.float64)
                    attn_weight_cov_for_lambda = attn_weight_cov.to(torch.float64)
                attn_lambda = compute_fro_proxy_lambda_terms(
                    x_cov=attn_cov_for_lambda,
                    weight_cov=attn_weight_cov_for_lambda,
                    low_bits=ptq_args.low_bits,
                    high_bits=ptq_args.high_bits,
                    high_len=high_length_hidden,
                )
                attn_obj_phys = (
                    attn_lambda["lambda_x"] * attn_cov_for_lambda
                    + attn_lambda["lambda_w"] * attn_weight_cov_for_lambda
                )
                attn_obj = (
                    attn_lambda["lambda_x"] * attn_cov.cpu()
                    + attn_lambda["lambda_w"] * attn_weight_cov
                )
                print(attn_lambda)
                if use_pre_fuse_lambda_compensation:
                    eval_attn, evec_attn_phys = perform_eigen_decomp(attn_obj_phys)
                    evec_attn = map_basis_to_fused_orthonormal(
                        evec_attn_phys, attn_ln_scale
                    )
                else:
                    eval_attn, evec_attn = perform_eigen_decomp(attn_obj)
            else:
                eval_attn, evec_attn = perform_eigen_decomp(attn_cov)

            # eigen decomposition of up proj
            mlp_cov = H_mlp[i] / (nbatches * seqlen)
            if ptq_args.basis_cov_mode == "wa_cov":
                mlp_weight_cov = fused_input_weight_cov(
                    [layers[i].mlp.gate_proj, layers[i].mlp.up_proj]
                )
                mlp_ln_scale = pre_fuse_ln_scale_map.get(i, {}).get("mlp")
                if mlp_ln_scale is None or int(mlp_ln_scale.numel()) != int(mlp_cov.shape[0]):
                    mlp_ln_scale = torch.ones(int(mlp_cov.shape[0]), dtype=torch.float64)
                if use_pre_fuse_lambda_compensation:
                    mlp_cov_for_lambda = diag_congruence_transform(
                        mlp_cov.cpu(), mlp_ln_scale, inverse=False
                    )
                    mlp_weight_cov_for_lambda = diag_congruence_transform(
                        mlp_weight_cov, mlp_ln_scale, inverse=True
                    )
                else:
                    mlp_cov_for_lambda = mlp_cov.cpu().to(torch.float64)
                    mlp_weight_cov_for_lambda = mlp_weight_cov.to(torch.float64)
                mlp_lambda = compute_fro_proxy_lambda_terms(
                    x_cov=mlp_cov_for_lambda,
                    weight_cov=mlp_weight_cov_for_lambda,
                    low_bits=ptq_args.low_bits,
                    high_bits=ptq_args.high_bits,
                    high_len=high_length_hidden,
                )
                mlp_obj_phys = (
                    mlp_lambda["lambda_x"] * mlp_cov_for_lambda
                    + mlp_lambda["lambda_w"] * mlp_weight_cov_for_lambda
                )
                mlp_obj = (
                    mlp_lambda["lambda_x"] * mlp_cov.cpu()
                    + mlp_lambda["lambda_w"] * mlp_weight_cov
                )
                print(mlp_lambda)
                if use_pre_fuse_lambda_compensation:
                    eval_mlp, evec_mlp_phys = perform_eigen_decomp(mlp_obj_phys)
                    evec_mlp = map_basis_to_fused_orthonormal(
                        evec_mlp_phys, mlp_ln_scale
                    )
                else:
                    eval_mlp, evec_mlp = perform_eigen_decomp(mlp_obj)
            else:
                eval_mlp, evec_mlp = perform_eigen_decomp(mlp_cov)

            # eigen decomposition of down proj
            eval_down_proj, evec_down_proj = perform_eigen_decomp(
                H_down_proj[i] / (nbatches * seqlen)
            )

            # eigen decomposition of value states
            value_cov = H_value[i] / (seqlen * nbatches)
            if ptq_args.basis_cov_mode == "wa_cov" and use_kv_wa_cov:
                value_weight_cov = build_o_proj_weight_cov_per_kv_head(
                    layers[i].self_attn.o_proj,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=kv_heads,
                    head_dim=head_dim,
                ).to(torch.float64).cpu()
                value_cov_64 = value_cov.to(torch.float64).cpu()
                value_obj = torch.zeros_like(value_cov_64, dtype=torch.float64)
                for hd in range(int(value_cov_64.shape[0])):
                    value_lambda = compute_fro_proxy_lambda_terms(
                        x_cov=value_cov_64[hd],
                        weight_cov=value_weight_cov[hd],
                        low_bits=ptq_args.low_bits,
                        high_bits=ptq_args.high_bits,
                        high_len=high_length_head,
                    )
                    value_obj[hd] = (
                        value_lambda["lambda_x"] * value_cov_64[hd]
                        + value_lambda["lambda_w"] * value_weight_cov[hd]
                    )
                eval_value, evec_value = perform_eigen_decomp(
                    value_obj,
                    per_head=True,
                    num_heads=value_obj.shape[0],
                )
            else:
                eval_value, evec_value = perform_eigen_decomp(
                    value_cov,
                    per_head=True,
                    num_heads=value_cov.shape[0],
                )

            # eigen decomposition of key states after rope embedding
            key_cov = H_key_pos[i] / (seqlen * nbatches)
            if ptq_args.basis_cov_mode == "wa_cov" and use_kv_wa_cov:
                key_cov_64 = key_cov.to(torch.float64).cpu()
                query_cov_for_k = (H_query_pos[i] / (seqlen * nbatches)).to(torch.float64).cpu()
                key_obj = torch.zeros_like(key_cov_64, dtype=torch.float64)
                for hd in range(int(key_cov_64.shape[0])):
                    key_lambda = compute_fro_proxy_lambda_terms(
                        x_cov=key_cov_64[hd],
                        weight_cov=query_cov_for_k[hd],
                        low_bits=ptq_args.low_bits,
                        high_bits=ptq_args.high_bits,
                        high_len=high_length_head,
                    )
                    key_obj[hd] = (
                        key_lambda["lambda_x"] * key_cov_64[hd]
                        + key_lambda["lambda_w"] * query_cov_for_k[hd]
                    )
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    key_obj,
                    per_head=True,
                    num_heads=key_obj.shape[0],
                )
            else:
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    key_cov,
                    per_head=True,
                    num_heads=key_cov.shape[0],
                )

            print(
                "up proj:",
                (eval_mlp[0:100].sum() / eval_mlp.sum()).item(),
                "down proj:",
                (eval_down_proj[0:100].sum() / eval_down_proj.sum()).item(),
                ", hidden_attn:",
                (eval_attn[0:100].sum() / eval_attn.sum()).item(),
                ", v_proj:",
                (eval_value[:, :32].sum(1) / eval_value.sum(1)).mean().item(),
                # ", kq_proj:",
                # (eval_kq[:, :32].sum(1) / eval_kq.sum(1)).mean().item(),
                ", k_proj_pos:",
                (eval_k_pos[:, :32].sum(1) / eval_k_pos.sum(1)).mean().item(),
            )

            basis_dict["config"] = "per_layer_rotation"
            basis_dict["layer." + str(i) + ".mlp"] = evec_mlp.cpu()
            basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()
            basis_dict["layer." + str(i) + ".self_attn"] = evec_attn.cpu()
            basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
            basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()

        torch.cuda.empty_cache()

        torch.save(
            basis_dict,
            basis_path,
        )

        torch.save(
            eval_dict,
            eval_path,
        )
    else:
        print(f"Basis rotations already exist at {basis_path}")


def perform_eigen_decomp(Cov_matrix, per_head=False, num_heads=0):
    # performs eigen decomposition and returns
    # the sorted eigen values and eigen vectors
    Cov_matrix = Cov_matrix.to(utils.DEV)
    if per_head:
        if Cov_matrix.ndim != 3:
            raise ValueError(
                f"Expected a 3D covariance tensor for per-head eigendecomposition, got shape {tuple(Cov_matrix.shape)}"
            )
        if num_heads == 0:
            num_heads = Cov_matrix.shape[0]
        if num_heads > Cov_matrix.shape[0]:
            raise ValueError(
                f"num_heads ({num_heads}) exceeds available heads in covariance tensor ({Cov_matrix.shape[0]})"
            )
        eval = []
        evec = []
        for hd in range(num_heads):
            H = Cov_matrix[hd]
            damp = 0.01 * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[-1]).to(device=H.device)
            H[diag, diag] = H[diag, diag] + damp
            X = torch.linalg.eigh(H.to(torch.float64))
            index = torch.argsort(X[0])
            eval.append(X[0][index])
            evec.append(X[1][:, index])
        eval = torch.stack(eval)
        evec = torch.stack(evec)
    else:
        H = Cov_matrix
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1]).to(device=H.device)
        H[diag, diag] = H[diag, diag] + damp
        X = torch.linalg.eigh(H.to(torch.float64))
        index = torch.argsort(X[0])
        eval = X[0][index]
        evec = X[1][:, index]

    return eval, evec


if __name__ == "__main__":
    model_args, training_args, ptq_args = process_args_ptq()
    get_basis(model_args, training_args, ptq_args)
