# seq2rel: A sequence-to-sequence approach for document-level relation extraction

[![ci](https://github.com/JohnGiorgi/seq2rel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JohnGiorgi/seq2rel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JohnGiorgi/seq2rel/branch/main/graph/badge.svg?token=RKJ7EV4WQK)](https://codecov.io/gh/JohnGiorgi/seq2rel)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johngiorgi/seq2rel/main/demo.py)

The corresponding code for our paper: A sequence-to-sequence approach for document-level relation extraction. Checkout our demo [here](https://share.streamlit.io/johngiorgi/seq2rel/main/demo.py)!

## Table of contents

- [seq2rel: A sequence-to-sequence approach for document-level relation extraction](#seq2rel-a-sequence-to-sequence-approach-for-document-level-relation-extraction)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Setting up a virtual environment](#setting-up-a-virtual-environment)
    - [Installing the library and dependencies](#installing-the-library-and-dependencies)
  - [Usage](#usage)
    - [Preparing a dataset](#preparing-a-dataset)
    - [Training](#training)
    - [Inference](#inference)

## Installation

This repository requires Python 3.8 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. If you need pointers on setting up a virtual environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#installing-via-pip).

### Installing the library and dependencies

If you _do not_ plan on modifying the source code, install from `git` using `pip`

```bash
pip install git+https://github.com/JohnGiorgi/seq2rel.git
```

Otherwise, clone the repository and install from source using [Poetry](https://python-poetry.org/):

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

Datasets are tab-separated files, where each example is contained on its own line. The first column contains the text, and the second column contains the relation. Relations themselves must be serialized to strings.

Take the following example, which expresses a _gene-disease association_ (`"@GDA@"`) between _ESR1_ (`"@GENE@"`) and _schizophrenia_ (`"@DISEASE@`")

```
Variants in the estrogen receptor alpha (ESR1) gene and its mRNA contribute to risk for schizophrenia. estrogen receptor alpha ; ESR1 @GENE@ schizophrenia @DISEASE@ @GDA@
```

For convenience, we provide a second package, [seq2rel-ds](https://github.com/JohnGiorgi/seq2rel-ds), which makes it easy to generate data in this format for various popular corpora. See our paper for more details on serializing relations.

### Training

To train the model, use the [`allennlp train`](https://docs.allennlp.org/main/api/commands/train/) command with [one of our configs](https://github.com/JohnGiorgi/seq2rel/tree/main/training_config) (or write your own!)

For example, to train a model on the [BioCreative V CDR task corpus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/), first, preprocess this data with [seq2rel-ds](https://github.com/JohnGiorgi/seq2rel-ds)

```bash
seq2rel-ds cdr main "path/to/preprocessed/cdr"
```

Then, call `allennlp train` with the [CDR config we have provided](https://github.com/JohnGiorgi/seq2rel/tree/main/training_config/cdr.jsonnet)

```bash
train_data_path="path/to/preprocessed/cdr/train.tsv" \
valid_data_path="path/to/preprocessed/cdr/valid.tsv" \
dataset_size=500 \
allennlp train "training_config/cdr.jsonnet" \
    --serialization-dir "output" \
    --include-package "seq2rel" 
```

During training, models, vocabulary, configuration, and log files will be saved to the directory provided by `--serialization-dir`. This can be changed to any directory you like. 

### Inference

To use the model to extract relations, import `Seq2Rel` and pass it some text

```python
from seq2rel import Seq2Rel
from seq2rel.common import util

# Pretrained models are stored on GitHub and will be downloaded and cached automatically.
# See: https://github.com/JohnGiorgi/seq2rel/releases/tag/pretrained-models.
pretrained_model = "gda"

# Models are loaded via a simple interface
seq2rel = Seq2Rel(pretrained_model)

# Flexible inputs. You can provide...
# - a string
# - a list of strings
# - a text file (local path or URL)
input_text = "Variations in the monoamine oxidase B (MAOB) gene are associated with Parkinson's disease (PD)."

# Pass any of these to the model to generate the raw output
output = seq2rel(input_text)
output == ["monoamine oxidase b ; maob @GENE@ parkinson's disease ; pd @DISEASE@ @GDA@"]

# To get a more structured (and useful!) output, use the `extract_relations` function
extract_relations = util.extract_relations(output)
extract_relations == [
  {
    "GDA": [
      ((("monoamine oxidase b", "maob"), "GENE"),
      (("parkinson's disease", "pd"), "DISEASE"))
    ]
  }
]
```

See the list of available `PRETRAINED_MODELS` in [seq2rel/seq2rel.py](seq2rel/seq2rel.py)

```bash
python -c "from seq2rel import PRETRAINED_MODELS ; print(list(PRETRAINED_MODELS.keys()))"
```

