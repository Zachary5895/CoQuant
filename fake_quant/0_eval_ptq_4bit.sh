# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

BASIS_COV_MODE=${BASIS_COV_MODE:-wa_cov}

torchrun --nnodes=1 --nproc_per_node=1 fake_quant/ptq.py \
--input_model /model/llama3.2/1b \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 4 \
--a_bits 4 \
--k_bits 4 \
--v_bits 4 \
--high_bits 8 \
--low_bits 2 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64 \
--high_fraction 0.125 \
--low_fraction 0.0 \
--rotate_mode "mix" \
--optimized_rotation_path output/R-high-0.125-low-0.0-sparse-0.0-qwen2.5-0.5b.bin \
--optimized_basis_path output/U-${BASIS_COV_MODE}-wikitext-512-qwen2.5-0.5b.bin \
--rotation_granularity 'per_layer' \
--rotate \
