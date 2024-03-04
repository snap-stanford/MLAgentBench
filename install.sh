#/bin/bash

# auto-gpt
# pip install -r Auto-GPT/requirements.txt

# crfm api
# pip install crfm-helm

# ML dependencies
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install -r requirements.txt
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
python -m pip install typing-inspect==0.8.0 typing_extensions==4.5.0
python -m pip install pydantic -U
python -m pip install -U jax==0.4.6 jaxlib==0.4.6+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install -U numpy
python -m pip install --force-reinstall charset-normalizer==3.1.0