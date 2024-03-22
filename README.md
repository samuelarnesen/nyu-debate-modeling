#  NYU Debate Modeling Project

Note: Given the current state of this project, this README will just give an overview of the code structure. It is not an introduction to the overall effort.

## Setup

### Basic Setup
1. Pull down this package using `git clone`
2. Install the dependencies using `pip install -r requirements.txt`
3. Run the setup script `bash bash_scripts/setup.sh`
4. Create an `.env` file that, at a minimum, contains an entry called `SRC_ROOT` that contains the file path of the root of this code base.
4. If you want to use the OpenAI models, add `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` to your `.env`. Similarly, if you want to use a model that requires access to one of Meta's models (e.g. Llama 2), add `META_ACCESS_KEY` to your `.env` file as well.

### HPC Setup
1. Follow the HPC's guide to setup a Singularity container. Remember to install the dependencies in the virtual environment in that container.
2. Run `bash bash_scripts/hpc_setup.sh` from inside the Singularity container

**Note: You will need to write an SBATCH script in order to run on the HPC. While doing so, remember to specifically request an A100 since the other GPU types do not support FlashAttention.**

## Tests

To run the tests, run `bash bash_scripts/basic_tests.sh`. To run an individual test, run `bash bash_scripts/operational_test.sh [TEST_NAME]`. To test the training code, run `bash bash_scripts/train_tests.sh`

## Code Structure

### High-Level Summary

One can use this package to train models or to run experiments (aka validation or inference). The key modules are as follows:

1. **Data:** This module provides a unified interface for loading, parsing, and interacting with the debate datasets. 
2. **Prompts:** This modules contains a configuration file and parser where you define the prompts you expose to the debaters and judge. It interacts with the `data` module to fill out the prompts with actual values.
3. **Models:** This module contains a unified interface for the resources that can generate text. Each model can wrap around an open LLM (e.g. Llama 2), an API (e.g. OpenAI's GPT4), or any other mechanism for generating text (e.g. it is convenient, for testing purposes, to have a model that generates random text). 
4. **Debate:** This module defines the core debate-specific abstractions. That includes the abstraction for an agent (i.e. a `Debater` or `Judge`), a `Transcript`, a speech order, and a `DebateRound`. The agents maintain an internal transcript and wrap around a `Model` in order to generate text.
5. **Experiment:** This contains the logic for running experiments. A loader read a config file and creates a series of `DebateRound` objects, whose results are then tallied, logged, and displayed.
6. **Train:** This modules contains all the logic for training a model. It interacts with the `prompt`, `data`, `models` and `debate` modules to consistency across input and output formats.
7. **Scripts:** This is the entrance point for running experiments and training models.

### Scripts
The primary entrance points are in the `scripts` directory. Here is what each of the scripts are intended to do:
* **load_model.py**: This just loads the model and saves it to your machine. This is a separate job so that one can download a (potentially large) model without having to request GPU time. Rarely used.
* **run_debate.py**: This kicks off a series of debate rounds. The configuration for those rounds (number of debaters, models used, number of speeches, use of scratchpad, decoding strategy, batching, etc.) is set in `experiments/configs`. Example Usage: `python3 ./scripts/run_debate.py --num_iters=50 --log_level='DEBUG' --configuration='13B Validation'`
* **run_dpo.py**: Kicks off a DPO training run. The configuration for the training run is set in `train/configs/dpo_config.yaml`. Example usage: `python3 ./scripts/run_dpo.py --configuration='13B - Alpaca - Basic' --dataset='/path/to/bon/dataset/prefix-of-bon-files' --train_type='dpo'`
* **run_ppo.py**: Kicks off a PPO training run. The configuration for the training run is set in `train/configs/ppo_config.yaml`. Example usage: `python3 /home/spa9663/debate/scripts/run_ppo.py --configuration='Train - 13B - Alpaca' --train_type='ppo'`
* **run_sft.py**: Kicks off a SFT training run. The configuration for the training run is set in `train/configs/sft_config.yaml`. Example usage: `python3 ./scripts/run_sft.py --configuration='Train - 13B - Alpaca' --train_type='sft'`

### Data
This module handles the loading of different datasets. The data loaded in these datasets is used for the questions that are debated in the experiments as well as the training signal for the training scripts. The datasets are as follows:
* **Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments.
* **Quality**: These are the full set of hard questions from the QuALITY dataset.
* **Annotated Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments, with the opening speeches annotated with different tags pertaining to style and tone.
* **Scratchpad Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments, annotated alongside an automatically-generated scratchpad that records the quotes used.

### Experiments
This module controls the configuration for running experiments (aka debate rounds) as well as the metrics collection for those experiments. See `experiments/configs/example_experiment.yaml` for a detailed explanation of each potential configuration option.
* **annotator.py**: This runs a classification model over the transcripts of a debate so that we can identify stylistic features.
* **experiment_loader.py**: This reads from the configuration set in `experiments/configs` and constructs a set of `DebateRound` objects.
* **quotes_collector.py**: This handles the metrics collections for the metrics about quotation usage.
* **results_collector.py**: This handles the overall metric collection.

## Agents
This module controls the primary abstractions for actually running experiments. The main abstractions are as follows:
* **Debate Round**: This is a structure containing two debaters and a judge. Each debater or judge generates speeches and collect their own transcripts. The judge is the main coordinator for the round, so the primary role of the `DebateRound` object is to pass information (speeches) from the different debaters and to report the winner at the end of the round.
* **Judge**: This agent asks questions, renders verdicts, and decides who the next speaker is. It uses a `Model` to generate text. It receives a `SpeechFormat` argument that determines the order of speeches that it expects.
* **Debater**: This wraps around a `Model` class, which it uses to generate speeches when called. There are child `Debater` classes for Best-of-N rounds (when it has to generate multiple speeches), and human rounds (when it's replaying the real human debates, which is useful for testing the judge models) that all inherit from the main `Debater` class. It receives a `SpeechFormat` argument that determines the order of speeches that it expects.
* **Transcript**: Each participant (judge and debater) maintains a `Transcript` object that tracks the round so far. It can convert into either a `ModelInput` format that can be used by a `Model` or a string that can be written to the logs.
* **Speech Format**: This defines the order of speeches. It has default options for both debate and consultancy. These get passed to the debaters and judge so that they know whose turn it is to speak.

## Models
This module also contains the `Model` abstraction that is used to generate text. The models are as follows:
* **Deterministic Model**: This model just outputs the same text over and over again. This is useful when testing.
* **Human Model**: This model outputs real text from the corresponding human debate transcripts. It only works if one is sampling questions from the real human debates.
* **LLM Model**: This model generates text by invoking a model from Huggingface (yes I know the "M" in LLM stands for "model" but "ll_model" looked strange). It has child classes for different flavors of Huggingface models that have different input formats. At the moment, those two implementing classes are for Llama and Mistral.
* **Offline Model**: This model just repeats the text from a text transcript that it is provided. This is useful if one wants to re-evaluate the debate with a different judge than the one originally used.
* **OpenAI Model**: This model generates text by calling OpenAI.
* **Random Model**: This model generates random strings of text. It is useful when testing.
* **Served Model**: This model makes requests to a localhost destination where it expects a model to be hosted (this can speed up inference dramatically if the model is hosted using an inference engine).

## Prompts
This module controls the prompts that are used while conducting experiments or training. There is a parser for both normal prompts and 'dynamic prompts' (which are just prompts that change depending on certain attributes). The actual prompt language can be found in `prompts/configs`

## Train
This module controls the logic for finetuning the models using DPO, PPO, or supervised finetuning. The specific classes are:
* **DPO Trainer**: This is a class for training models using DPO
* **PPO Trainer**: This is a class for training models using DPO
* **Pretrain Trainer**: This is a class for pretraining a model. This is no longer used and was introduced to see if we could get the models to memorize the background text.
* **Probe Trainer**: This is a class for training a linear probe that sits on top of a Huggingface model. This was used only briefly and isn't as tested as the DPO, PPO, and SFT Trainers.
* **SFT Trainer**: This is a class for training a model to debate via supervised finetuning.
* **Row Converter**: This is a utility for converting a transcript into the format expected by the trainers. To do so, it interacts with the `debate`, `prompts`, and `data` abstractions.
* **Train Utils**: Additional utilities for loading models and datasets, specifically for training.


