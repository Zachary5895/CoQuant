import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from eval_utils.main import ptq_model
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
import numpy as np
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import random
from utils.parallel_utils import map_layers_to_multi_gpus
import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

log: Logger = utils.get_logger("CoQuant", "log.log")

@torch.no_grad()
def evaluate(model, tokenizer, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    results = {}

    testloader = data_utils.get_wikitext2(
        seed=args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    model.config.use_cache = use_cache


    if args.multigpu:
        map_layers_to_multi_gpus(model.model.layers)
        input_device = model.model.layers[0].device
        output_device = model.model.layers[-1].device
        assert input_device == output_device
        model.model.embed_tokens.to(input_device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(input_device)
        if hasattr(model, "visual"):
            model.visual = model.visual.to(input_device)
        model.model.norm.to(output_device)
        model.lm_head.to(output_device)
    else:
        input_device = utils.DEV
        model.to(utils.DEV)
    if args.tasks != "":
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
        model_args={}
        model_args['trust_remote_code'] = True
        #task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
        results = lm_eval.simple_evaluate(hflm, tasks=args.tasks, model_args=model_args, batch_size=args.lm_eval_batch_size)['results']
        print(results)

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        print(metric_vals)

    return results

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()
    seed_everything(ptq_args.seed)
    config = AutoConfig.from_pretrained(
        model_args.input_model,
    )
    
    if ptq_args.flash_attn:
        config._attn_implementation = "flash_attention_2"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    vision = False
    if "llama" in model_args.input_model.lower():
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype,
            config=config,
        )
    elif "qwen2" in model_args.input_model.lower() and "vl" not in model_args.input_model.lower():
        model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype,
            config=config,
        )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))
    model = ptq_model(ptq_args, model, model_args)
    print(model)
    model.seqlen = training_args.model_max_length

    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    if vision:
        tokenizer = AutoProcessor.from_pretrained(model_args.input_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
        )
    log.info("Complete tokenizer loading...")

    ptq_args.model_name = model_args.input_model.split('/')[-2] + model_args.input_model.split('/')[-1]
    results = evaluate(model, tokenizer, ptq_args)
    dist.barrier()


if __name__ == "__main__":
    train()
