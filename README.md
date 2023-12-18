#  NYU Debate Modeling Project

Note: Given the current state of this project, this README will just give an overview of the code structure. It is not an introduction to the overall effort.

## Setup

### Basic Setup
1. Pull down this package using `git clone`
2. Install the dependencies using `pip install -r requirements.txt`
3. Run the setup script `bash bash_scripts/setup.sh`
4. Create an `.env` file that, at a minimum, contains an entry called `SRC_ROOT` that contains the file path of the root of this code base.
4. If you want to use the OpenAI models, add `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` to your `.env`. Similarly, if you want to use a model that requires access to one of Meta's models (e.g. Llama 2), add `META_ACCESS_KEY` to your `.env` file as well.

**Note: This code has dependencies on the transcripts from the human debate project and (optionally) on an annotated version of that dataset, along with a classifier model that was trained on those annotations. Since those aren't all publicly released yet, they are not present in the repo.**

### HPC Setup
1. Follow the HPC's guide to setup a Singularity container. Remember to install the dependencies in the virtual environment in that container.
2. Run `bash bash_scripts/hpc_setup.sh` from inside the Singularity container

**Note: You will need to write an SBATCH script in order to run on the HPC. While doing so, remember to specifically request an A100 since the other GPU types do not support FlashAttention.**

**Note: There are still points in the code that point to specific places on either my local computer or on my partition of the HPC. You will need to update `scripts/script_utils.py` and all the config files in order to get things to actually run**.


## Tests

To run the tests, run `bash bash_scripts/basic_tests.sh`. To run an individual test, run `bash bash_scripts/operational_test.sh [TEST_NAME]`

## Code Structure

### Scripts
The primary entrance points are in the `scripts` directory. Here is what each of the scripts are intended to do:
* **load_model.py**: This just loads the model and saves it to your machine. This is a separate job so that one can download a (potentially large) model without having to request GPU time. Rarely used.
* **run_debate.py**: This kicks off a series of debate rounds. The configuration for those rounds (number of debaters, models used, number of speeches, use of scratchpad, decoding strategy, batching, etc.) is set in `experiments/configs`. Example Usage: `python3 ./scripts/run_debate.py --num_iters=50 --log_level='DEBUG' --configuration='13B Validation - Normal Quality'`
* **run_dpo.py**: Kicks off a DPO training run. The configuration for the training run is set in `train/configs/dpo_config.yaml`. Example usage: `python3 ./scripts/run_dpo.py --configuration='13B - Alpaca - Basic' --dataset='/path/to/bon/dataset/prefix-of-bon-files' --train_type='dpo'`
* **run_ppo.py**: Kicks off a PPO training run. The configuration for the training run is set in `train/configs/ppo_config.yaml`. Example usage: `python3 /home/spa9663/debate/scripts/run_ppo.py --configuration='Train - 13B - Alpaca' --train_type='ppo'`
* **run_sft.py**: Kicks off a SFT training run. The configuration for the training run is set in `train/configs/sft_config.yaml`. Example usage: `python3 ./scripts/run_sft.py --configuration='Train - 13B - Alpaca' --train_type='sft'`
* **run_split_debate.py**: Not currently used. This will be used when we do multi-stage debate rounds that involve branching (aka we want to replay up to a specific point, after which it splits into a new round).

### Experiments
This module controls the configuration for running experiments (aka debate rounds) as well as the metrics collection for those experiments. See `experiments/configs/example_experiment.yaml` for a detailed explanation of each potential configuration option.
* **annotator.py**: This runs the classification model over the transcripts of a debate so that we can identify stylistic features.
* **experiment_loader.py**: This reads from the configuration set in `experiments/configs` and constructs a set of `DebateRound` objects.
* **quotes_collector.py**: This handles the metrics collections for the metrics about quotation usage.
* **results_collector.py**: This handles the overall metric collection.

### Data
This module handles the loading of different datasets. The data loaded in these datasets is used for the questions that are debated in the experiments as well as the training signal for the training scripts. The datasets are as follows:
* **Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments.
* **Quality**: These are the full set of hard questions from the QuALITY dataset.
* **Annotated Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments, with the opening speeches annotated with different tags pertaining to style and tone.

## Agents
This module controls the primary abstractions for running experiments and interacting with models. The main abstractions are as follows:
* **Debate Round**: This is a structure containing two debaters and a judge. They iteratively generate speech and collect their own transcripts. The judge is the main coordinator for the round, so the primary role of the `DebateRound` object is to pass information (speeches) from the different debaters and to report the winner at the end of the round.
* **Judge**: This asks questions, renders verdicts, and decides who the next speaker is. It uses a `Model` to generate text. It receives a `SpeechFormat` argument that determines the order of speeches that it expects. There is a related judge for creating Best-of-N preference judgements that inherits from the main judge class.
* **Debater**: This wraps around a `Model` class, which it uses to generate speeches when called. There are child `Debater` classes for Best-of-N rounds (when it has to generate multiple speeches), offline rounds (when it doesn't actually need to generate new content and can rely on an existing transcript), and human rounds (when it's replaying the real human debates, which is useful for testing the judge models) that all inherit from the main `Debater` class. It receives a `SpeechFormat` argument that determines the order of speeches that it expects.
* **Transcript**: Each participant (judge and debater) maintains a `Transcript` object that tracks the round so far. It can convert into either a `ModelInput` format that can be used by a `Model` or a string that can be written to the logs.

This module also contains the `Model` abstraction that is used to generate text. The models are as follows:
* **Deterministic Model**: This model just outputs the same text over and over again. This is useful when testing.
* **Human Model**: This model outputs real text from the corresponding human debate transcripts. It only works if one is sampling questions from the real human debates.
* **Llama Model**: This model generates text by invoking a version of Llama2.
* **Offline Model**: This model just repeats the text from a text transcript that it is provided. This is useful if one wants to re-evaluate the debate with a different judge than the one originally used.
* **OpenAI Model**: This model generates text by calling OpenAI.
* **Random Model**: This model generates random strings of text. It is useful when testing.

## Prompts
This module controls the prompts that are used while conducting experiments or training. There is a parser for both normal prompts and 'dynamic prompts' (which are just prompts that change depending on certain attributes). The actual prompt language can be found in `prompts/configs`

## Train
This module controls the logic for finetuning the models using DPO, PPO, or supervised finetuning. 


