#  NYU Debate Modeling Project

Note: Given the current state of this project, this README will just give an overview of the code structure. It is not an introduction to the overall effort.

## Setup

### Basic Setup
1. Pull down this package using `git clone`
2. Install the dependencies using `pip install -r requirements.txt`
3. Create an `.env` file with the following variables:
	`SRC_ROOT`: Contains the file path of the root of this code base
	`INPUT_ROOT`: Contains the file path to the directory where the dataset files are located
	`OPENAI_API_KEY`: OpenAI key (Optional, if one is using OpenAI for judging)
	`OPENAI_ORGANIZATION`: OpenAI organization (Optional, if one is using OpenAI for judging)
	`META_ACCESS_KEY`: Meta huggingface token, (Optional, if one is downloading a Llama model via HF)
	`ANTHROPIC_API_KEY`: Anthropic access key (Optional, if one is using Anthropic for judging)

### HPC Setup
1. Follow the HPC's guide to setup a Singularity container. Remember to install the dependencies in the virtual environment in that container.
2. Run `bash bash_scripts/hpc_setup.sh` from inside the Singularity container

## Tests

To run the tests, run `bash bash_scripts/basic_tests.sh`. To run an individual test, run `bash bash_scripts/operational_test.sh [TEST_NAME]`. To test the training code, run `bash bash_scripts/train_tests.sh`

### Scripts
The primary entrance points are in the `scripts` directory. Here is what each of the scripts are intended to do:
* **run_debate.py**: This kicks off a series of debate rounds. The configuration for those rounds (number of debaters, models used, number of speeches, use of scratchpad, decoding strategy, batching, etc.) is set in `experiments/configs`. Example Usage: `python3 ./scripts/run_debate.py --num_iters=50 --log_level='DEBUG' --configuration='Test'`
* **run_iterative_dpo.py**: Kicks off a DPO training run. The configuration for the training run is set in `train/configs/dpo_config.yaml`. Example usage: `python3 ./scripts/run_iterative_dpo.py --configuration='Test'`
* **run_sft.py**: Kicks off a SFT training run. The configuration for the training run is set in `train/configs/sft_config.yaml`. Example usage: `python3 ./scripts/run_sft.py --configuration='Test'`

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

### Data
This module handles the loading of different datasets. The data loaded in these datasets is used for the questions that are debated in the experiments as well as the training signal for the training scripts. The datasets are as follows:
* **Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments.
* **Quality**: These are the full set of hard questions from the QuALITY dataset.

### Debate
This module controls the primary abstractions for actually running experiments. The main abstractions are as follows:
* **Debate Round**: This is a structure containing two debaters and a judge. Each debater or judge generates speeches and collect their own transcripts. The judge is the main coordinator for the round, so the primary role of the `DebateRound` object is to pass information (speeches) from the different debaters and to report the winner at the end of the round.
* **Judge**: This agent asks questions, renders verdicts, and decides who the next speaker is. It uses a `Model` to generate text. It receives a `SpeechFormat` argument that determines the order of speeches that it expects. There is a child `BranchedJudge` class for the use in branched rounds.
* **Debater**: This wraps around a `Model` class, which it uses to generate speeches when called. There is a child `BestOfNDebater` class for Best-of-N rounds (when it has to generate multiple speeches). It receives a `SpeechFormat` argument that determines the order of speeches that it expects.
* **Transcript**: Each participant (judge and debater) maintains a `Transcript` object that tracks the round so far. It can convert into either a `ModelInput` format that can be used by a `Model` or a string that can be written to the logs.
* **Speech Format**: This defines the order of speeches. It has default options for both debate and consultancy. These get passed to the debaters and judge so that they know whose turn it is to speak.

### Experiments
This module controls the configuration for running experiments (aka debate rounds) as well as the metrics collection for those experiments. See `experiments/configs/example_experiment.yaml` for a detailed explanation of each potential configuration option.
* **experiment_loader.py**: This reads from the configuration set in `experiments/configs` and constructs a set of `DebateRound` objects.
* **quotes_collector.py**: This handles the metrics collections for the metrics about quotation usage.
* **results_collector.py**: This handles the overall metric collection.

### Models
This module also contains the `Model` abstraction that is used to generate text. The models are as follows:
* **Deterministic Model**: This model just outputs the same text over and over again. This is useful when testing.
* **LLM Model**: This model generates text by invoking a model from Huggingface (yes I know the "M" in LLM stands for "model" but "ll_model" looked strange). It has child classes for different flavors of Huggingface models that have different input formats. At the moment, those two implementing classes are for Llama and Mistral models. This also contains logic for implementing a Linear Probe judge (aka a judge that adds a linear layer on top of the activations of another model), however this is less tested.
* **Offline Model**: This model just repeats the text from a text transcript that it is provided. This is useful if one wants to re-evaluate the debate with a different judge than the one originally used.
* **OpenAI Model**: This model generates text by calling OpenAI.
* **Anthropic Model**: This model generates text by calling Anthropic's Claude model.
* **Random Model**: This model generates random strings of text. It is useful when testing.
* **Served Model**: This model makes requests to a localhost destination where it expects a model to be hosted (this can speed up inference dramatically if the model is hosted using an inference engine).
* **Human Model**: This model outputs real text from the corresponding human debate transcripts. It only works if one is sampling questions from the real human debates. (Not recommended)

### Prompts
This module controls the prompts that are used while conducting experiments or training. There is a parser for both normal prompts and 'dynamic prompts' (which are just prompts that change depending on certain attributes). The actual prompt language can be found in `prompts/configs`

### Train
This module controls the logic for finetuning the models using DPO, PPO, or supervised finetuning. The specific classes are:
* **DPO Trainer**: This is a class for training models using DPO
* **PPO Trainer**: This is a class for training models using DPO
* **SFT Trainer**: This is a class for training a model to debate via supervised finetuning.
* **Row Converter**: This is a utility for converting a transcript into the format expected by the trainers. To do so, it interacts with the `debate`, `prompts`, and `data` abstractions.
* **Train Utils**: Additional utilities for loading models and datasets, specifically for training.

## Potential Uses

### Running a Tournament
1. Create a new config entry under `experiments/configs`. If you're running this locally, add the entry under `test_experiment.yaml` and if you're running it remotely, add it under `standard_experiment.yaml`. 
2. To kick off the new tournament, run `python python3 ./scripts/run_debate.py --num_iters=Number-of-Iterations --configuration='Your-New-Configuration-Name`. If you're running it locally, you'll need to a `--test` flag so it knows to look in the right place.

### Adding a New Dataset
You might want to create a new dataset if you're using a data source other than QuALITY. Here are the steps to add a new dataset:
1. Create a new directory under `data/datasets/` and add your file(s) there.
2. Under the `data` directory, create a python file called `[your_dataset_name]_loader.py`. In that file, you will define two classes, a `[YourDatasetName]Loader` and `[YourDatasetName]Dataset`. The loader should just parse your dataset and pass it in to the dataset constructor. The dataset itself will split out the data into the rows that you want, following the interface defined in `data/dataset.py`. The file `data/quality_loader.py` is also a good example that one can follow (although it has some extra deduplication logic you may not need).
3. Under `data/dataset.py`, add a new enum corresponding to your new dataset type.
4. Under `data/loader_utils.py`, add a new conditional to support creating your new dataset.
5. Under `data/__init__.py`, import your new dataset.

Now you should be good to reference your new dataset from a train or experiment config file.

### Adding New Prompts
1. Go to `prompts/configs/prompts.yaml` and add a new entry corresponding to your new set of prompts. Feel free to follow the examples in that file -- the two sets of existing prompts are called `Debate Prompt` and `Consultancy Prompt`.
2. If your new prompt uses any unique names that do not already exist in the existing prompts (the 'names' are the titles of each individual sub-entry such as `overall_system` or `pre_opening_speech`), then go to `prompts/parser.py` and add that title as a new enum under `PromptTag`.
3. If you require filling in a new kind of value (currently, we support filling out prompts with a debater name, opponent name, position, opponent position, topic/question, and the background text), then add that new kind of value under `PromptConfig` in `prompts/parser.py`. This new value will be inserted into the prompts as long as the exact key is referenced inside angular brackets (e.g. if you reference `<NAME>` in your prompts, then `<NAME>` will be replaced with the value associated with `name` in the PromptConfig.) 

### Training a New Model
1. Depending on how you want to train your model (SFT, DPO, PPO, etc.), add a new entry in the associated config that can be found under `train/configs/[dpo/ppo/sft]_config.yaml`.
2. Kick off your training run by running the associated script in the `scripts` directory (e.g. `python scripts/run_sft.py --config="YourNewConfigName"`).

### Creating New Speech Orders
You might want to create new speech order if you do not want the speeches to be delivered in the same format as we have previously set.
1. Under `debate/speech_format.py`, create a new method in the `SpeechFormat` object, following the example of `default_debate_format()` and `default_judge_format()`. 
2. Also under `debate/speech_format.py`, create a new enum under `SpeechFormatStructure.`
3. Also under `debate/speech_format.py`, create a new enum under `SpeechFormatType.`

Now you can reference your new speech orders in your configs by using the name you gave under `SpeechFormatType`.

### Using a new open-weight LLM
We currently support generating speeches using Mistral or Llama. If you want to add a different type from Huggingface, you should do the following:
1. Under `models/llm_model.py`, create a new class that inherits from `LLModel`. At a minimum, it just needs to define the instruction prefixes and suffixes. See the `LlamaModel` class as a good example. It also should implement a `copy()` method.
2. Also under `models/llm_model.py`, create a new enum under `LLMType`.

Now you should be free to reference your new model type in your configs by using the name you defined as the `LLMType` enum.



