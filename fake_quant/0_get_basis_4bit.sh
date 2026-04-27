# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

BASIS_COV_MODE=${BASIS_COV_MODE:-wa_cov}

python fake_quant/get_basis.py \
--input_model /model/llama3.2/1b \
--output_rotation_path "output" \
--model_max_length 2048 \
--down_proj_blocksize 256 \
--high_fraction 0.125 \
--low_fraction 0.0 \
--rotation_granularity "per_layer" \
--rotate_mode "mix" \
--nsamples 512 \
--calib_dataset "wikitext" \
--basis_cov_mode "${BASIS_COV_MODE}" \
--sparse_fraction 0.0 \
--low_bits 4 \
--high_bits 8 \
--use_pre_fuse_lambda_compensation 0 \
--use_kv_wa_cov 0
