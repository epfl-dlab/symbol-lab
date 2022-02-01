
# Outline
The README contains a description of the **symbol-lab** environments, the models as well as instructions for running the code.

- Section [1](1.-project-description) contains a description of the project.
- Section [2](environments) talks about our synthetic environments.
- Section [3](models) describes the implemented models.
- Section [4](setup-instructions) presents the installation instructions and how to train the model.
- Section [5](hyperparameter-search) shows how to do a hyperparameter search for any model using our codebase.
- Section [6](unit-tests) presents the test cases and how to run them and ensure that the modules work as expected.

The codebase is built upon [PytorchLightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/). If you have not used them before, please take a look at the respective documentation so you can get to speed when running the code from this repo.

## 1. Project Description
Using continuous latent variables has been ubiquitous in deep learning; however, recent developments in discrete optimization have paved the path for exploiting the benefits of such representations, not only in modalities such as image or speech but particularly in the domain of language, which is inherently discrete.

Our goal to learn discrete representation of text in a self-supervised way. To that aim, we propose **symbol-lab**, an environment for testing the models’ ability to encode input into a symbolic (discrete) compositional representation. With a structure that allows for fine-grained control over the difficulty of the task, and an arbitrary amount of training data, symbol-lab aids us in uncovering the strengths and weaknesses of existing methods and facilitates the development of new ones. The learning methods/models that we are working with/on are listed in Section [3](models).


<!--- The model is an autoencoder where the encoder and decoder are themselves attentional encoder-decoders.
We form a sequence of discrete latents (tokens) in the bottleneck and try to reconstruct the input from the latent space.
We will use distant teacher forcing on the bottleneck to constrain the discrete representation to be in the form of (potentially more than 1) *entity-relation-entity* triples; thus, if the model is successful, we can automatically extract knowledge bases in the form of triples which has its broad applications.

To that end, we devise several synthetic environments and build upon them to gradually add to the complexity. Finally, we train our model end-to-end on a large corpus and compare it to the baselines.-->

## 2. Symbol-lab
<!--- As was mentioned above, we plan to increment the complexity from a simple grid of symbols which we call *Symbol-lab* to the full-blown language dataset consisting of a corpus to be reconstructed and a knowledge base in the form of triples to guide the discrete bottleneck. We will now discuss Symbol-lab and several other synthetic environments as it is a work in progress. The point of synthetic environments is having the ultimate fine-grained control over all aspects of the generated data distribution and thus the task's difficulty level. We can induce spurious correlations, introduce interventions in the test set and easily test the model's robustness and generalization abilities to problems it has not seen during training. Hence symbol-lab aids us in uncovering the strengths and weaknesses of existing methods and facilitates the development of new ones.
-->
Currently, **symbol-lab** supports two environments:
- **Grid Dataset**
- **Controlled decoder** operating on **structured discrete latent**

Below a short description of scenarios supported by each dataset is given, and you can find more details about each environment in datasets [readme](discrete_bottleneck/datamodule/symbol_lab_readme.md).

Note that these datasets inherit PyTorch's `Dataset` class. Therefore, you could either use **symbol-lab** through the PyTorch lightning datamodule or as standalone datasets in your code. You can refer to our usage example in this [notebook](notebooks/dataset_playground.ipynb).
### 2.1. Grid Dataset
Grid Dataset consists of samples representing grids of cells that can be occupied by different types of objects or can be left empty. Each sample of this dataset represents one random grid. Grid dataset supports two scenarios:
- **ConstantGrid**: In this dataset, all samples are grids of the same size and have the same number of objects.
- **VariableElementCountGrid**: In this dataset, all samples are grids of the same size, but the number of objects present in any sample grid can be anywhere between a predefined minimum and a maximum.

You can find the corresponding dataset classes with more details in the comments [here](discrete_bottleneck/datamodule/grid_datasets.py).

### 2.2. Structured Discrete Latent
Unlike Grid dataset in which the latents were not explicitly available nor they were used to generate the samples, in this dataset, we explicitly specify both the structure of the discrete latent $Z$ and the decoder's ($f$) properties. A sample of this dataset would be $f(z)$ where $z$ is sampled from the possible discrete space $Z$. With this dataset we can evaluate the performance of different models under any combination of the following scenarios:
Decoder $f$ can be:
- Stochastic vs. Deterministic
- Invertible vs. Non-invertible
- Outputs a single real number or a sequence

Discrete latent  $Z$ can be:
- A single discrete variable with 2 possible values
- A single discrete variable with $m>2$ possible values
- A sequence of discrete variables each with $m>2$ possible values

You can find the corresponding dataset classes with more details in the comments [here](discrete_bottleneck/datamodule/controlled_decoder_datasets.py).
## 3. Models

This repository (currently) implements three models:
- A sequence to sequence model with a continuous latent representation in the bottleneck ([Sutskever et al. 2014](https://arxiv.org/abs/1409.3215))
- A version of seq2seq that uses the quantization technique from [VQ-VAE](https://arxiv.org/abs/1711.00937) to obtain a discrete representation at the output of the encoder.
- Our adaptation of the discrete sequence autoencoder proposed by [SEQ^3](https://arxiv.org/abs/1904.03651).

Note, that both baselines encode the input sequence to one high-dimensional single vector representation (either discrete or continuous), but [SEQ^3](https://arxiv.org/abs/1904.03651) encodes the source text into a sequence of tokens/words.

## 4. Setup Instructions

- Clone this repository
- Run `docker build -t symbol_lab .` (have a folder with your public ssh key in the same directory; it will be used for login in to the container)
- Choose your preferred port for running a jupyter server (9991, for instance) and run the following:
    - `port=9991`
    - `docker run -ti --rm --gpus all -p $port:$port --name symbol_lab_container symbol_lab bash`
    - `cd / && jupyter lab --ip 0.0.0.0 --port 9991` (copy the url for the jupyter server host)
- `Detach from the container with Ctrl+P, then Ctrl-Q`
- Create an ssh tunnel between the local and the remote machine using: `ssh -f YOURUSERNAME@your_machine_address -L PORT:localhost:PORT -N` and enter your password when prompted.
- Paste the URL you copied earlier into your browser.

### Training
Login to your weights&biases (wandb) account, copy your `WANDB_API_KEY`, and set the environment variable. If you're in a notebook, `%env WANDB_API_KEY=YOUR_KEY`. If you're in the terminal, `export WANDB_API_KEY=YOUR_KEY`
The entry point for this codebase is `run.py` in the home directory of the package. To start training, type any of the following to train the corresponding model:

- `python3 run.py mode=train model=seq2seq`
- `python3 run.py mode=train model=vqvae`
- `python3 run.py mode=train model=seq3`

----------------------------
## 5. Hyperparameter Search

Use `hydra`'s `multirun` option for a manual sweep over hyperparameters. For instance let's say we want to train the `seq2seq` model with several learning rates and number of layers on the grid dataset with different number of objects when the number of rows and columns is fixed. We can use the following line (to either run that many jobs locally, or if configured properly, through SLURM)

```
python3 run.py mode=train model=seq2seq model.lr=0.001,0.0001 model.n_layers=2,4 datamodule=grid datamodule.num_rows=3 datamodule.num_columns=4 datamodule.num_objects_to_place=3,4,5,6 --multirun
```
If you are running your jobs locally or working on a cluster that is **not** managed by SLURM, refer to the following.
https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

To do hp optimization, `hydra` has several options: Optuna, Nevergrad, Ax Sweeper (later).

----------------------------
## 6. Unit Testing

- `python tests/test_dataset.py`
- `python tests/test_seq2seq_model.py`
- `python tests/performance_seq2seq_model.py`
