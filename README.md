![build](https://github.com/JohnGiorgi/seq2rel/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/JohnGiorgi/seq2rel/branch/master/graph/badge.svg)](https://codecov.io/gh/JohnGiorgi/seq2rel)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue)

# seq2rel

A Python package that makes it easy to use sequence-to-sequence (seq2seq) learning for information extraction.

## Installation

This repository requires Python 3.8 or later. The preferred way to install is via pip:

```
pip install seq2rel
```

If you need pointers on setting up an appropriate Python environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#installing-via-pip).

### Installing from source

You can also install from source. 

Using `pip`:

```
pip install git+https://github.com/JohnGiorgi/seq2rel
```

Using [Poetry](https://python-poetry.org/):

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

Coming soon.

### Training

Coming Soon.

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
>>> ['<ADE> ciprofloxacin <DRUG> renal insufficiency <EFFECT> </ADE>']
```

See the list of available `PRETRAINED_MODELS` in [seq2rel/seq2rel.py](seq2rel/seq2rel.py)

```bash
python -c "from seq2rel import PRETRAINED_MODELS ; print(list(PRETRAINED_MODELS.keys()))"
```

