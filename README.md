# seq2rel

[![ci](https://github.com/JohnGiorgi/seq2rel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JohnGiorgi/seq2rel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JohnGiorgi/seq2rel/branch/main/graph/badge.svg?token=RKJ7EV4WQK)](https://codecov.io/gh/JohnGiorgi/seq2rel)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johngiorgi/seq2rel/main/demo.py)


A Python package that makes it easy to use sequence-to-sequence (seq2seq) learning for information extraction. Checkout our demo [here](https://share.streamlit.io/johngiorgi/seq2rel/main/demo.py)!

## Installation

This repository requires Python 3.8 or later. The preferred way to install is via pip:


```bash
pip install git+https://github.com/JohnGiorgi/seq2rel.git
```

If you need pointers on setting up an appropriate Python environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#installing-via-pip).

### Installing from source

You can also install from source using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/JohnGiorgi/seq2rel
cd seq2rel

# Install the package with poetry
poetry install
```

## Usage

### Preparing a dataset

Datasets are tab-seperated files, where each example is contained on its own line. The first column contains the text, and the last column contains the relation. Relations themselves must be serialized to strings.

Take the following example, which denotes an _adverse drug event_ (`"@ADE@"`) between the drug _benzodiazepine_ (`"@DRUG@"`) and the effect _coma_ (`"@EFFECT@`")

```
A review of the literature showed no previous description of this pattern in benzodiazepine coma.	@ADE@ benzodiazepine @DRUG@ coma @EFFECT@ @EOR@
```

For convenience, we provide a second package, [seq2rel-ds](https://github.com/JohnGiorgi/seq2rel-ds), which makes it easy to generate data in this format for various popular corpora.

### Training

To train the model, use the [`allennlp train`](https://docs.allennlp.org/main/api/commands/train/) command with [one of our configs](https://github.com/JohnGiorgi/seq2rel/tree/main/training_config) (or write your own!)

For example, to train a model on the [Adverse Drug Event (ADE) corpus](https://github.com/trunghlt/AdverseDrugReaction/tree/master/ADE-Corpus-V2), first preprocess this data with [seq2rel-ds](https://github.com/JohnGiorgi/seq2rel-ds)

```bash
seq2rel-ds preprocess ade "path/to/preprocessed/ade"
```

Then, call `allennlp train` with the [ADE config we have provided](https://github.com/JohnGiorgi/seq2rel/tree/main/training_config/transformer_copynet_ade.jsonnet)

```bash
allennlp train "training_config/transformer_copynet_ade.jsonnet" \
    --serialization-dir "output" \
    --overrides "{'train_data_path': 'path/to/preprocessed/ade/train.tsv'}" \
    --include-package "seq2rel" 
```

The `--overrides` flag allows you to override any field in the config with a JSON-formatted string, but you can equivalently update the config itself if you prefer. During training, models, vocabulary, configuration, and log files will be saved to the directory provided by `--serialization-dir`. This can be changed to any directory you like. 

#### Hyperparameter tuning

Coming soon.

### Inference

To use the model as a library, import `Seq2Rel` and pass it some text (it accepts both strings and lists of strings)

```python
from seq2rel import Seq2Rel

# Pretrained models stored in GitHub. Downloaded and cached automatically. This model is ~500mb.
pretrained_model = "ade"

# Models are loaded via a dead-simple interface
seq2rel = Seq2Rel(pretrained_model)

# Extremely flexible inputs. User can provide...
# - a string
# - a list of strings
# - a text file (local path or URL)
input_text = "Ciprofloxacin-induced renal insufficiency in cystic fibrosis."

seq2rel(input_text)
>>> ['ciprofloxacin @DRUG@ renal insufficiency @EFFECT@ @ADE@']
```

See the list of available `PRETRAINED_MODELS` in [seq2rel/seq2rel.py](seq2rel/seq2rel.py)

```bash
python -c "from seq2rel import PRETRAINED_MODELS ; print(list(PRETRAINED_MODELS.keys()))"
```

