## Installation

USE PYTHON 3.11 OTHERWISE YOU MAY BE UNABLE TO GET ALL PACKAGES.

If you wish to install from a new Python environment, you can install the latest versions of the following packages; the following order listed resulted in the least issues with installation:

- matplotlib
- datasets
- sentence-transformers
- unsloth
- vllm==0.8.2
- [NPEET (must be downloaded from source)](https://github.com/gregversteeg/NPEET)

If for some reason the versions of the various packages are incompatible, you may have to uninstall and reinstall them. This will likely include PyTorch, which should be running on a CUDA-enabled version - you may have to uninstall and install the correct version (and then reinstall xformers) if this doesn't happen automatically.

NPEET must be downloaded manually from [the NPEET GitHub](https://github.com/gregversteeg/NPEET) before being installed in the environment. 

As Della does not provide Internet access during Slurm jobs, you must then download the weights for the models you wish to use. We have provided several examples which can be downloaded using the ```cache_for_offline.py``` file. Alternatively, they can also be individually downloaded manually through the HuggingFace CLI.

For the semantic MI reward, we require a separate semantic embedding to be downloaded. We have used the [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) model in our own experiments, but this is interchangeable with any other semantic embedding model. After downloading models, make sure that the variables for model filenames point to the correct places.

# TODO: change username to environment variable as to make saving to your own user directory a lot easier.