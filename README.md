# CoQuant: Joint Weight-Activation Subspace Projection for Mixed-Precision LLMs

## Building environment
1. Create conda environment
```
conda create -n "coquant" python=3.12.0
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Install fast-hadamard-transform library from [here](https://github.com/Dao-AILab/fast-hadamard-transform)
```
git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
```
## Running code
we show the example of how to quantize llama3.2-1b with CoQuant and baseline-ResQ
### CoQuant
1. Get projection matrices
```
BASIS_COV_MODE=wa_cov bash 0_get_basis_4bit.sh
```
2. Quantize and evaluate the model
```
BASIS_COV_MODE=wa_cov bash 0_eval_ptq_4bit.sh
```

### Resq
1. Get projection matrices
```
BASIS_COV_MODE=a_cov bash 0_get_basis_4bit.sh
```
2. Quantize and evaluate the model
```
BASIS_COV_MODE=a_cov bash 0_eval_ptq_4bit.sh
```

## Acknowledgements

Our implementation is developed based on the official implementation of [ResQ](https://github.com/utkarsh-dmx/project-resq). We sincerely thank the authors for making their code publicly available.