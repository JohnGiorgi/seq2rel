
seq2rel
=======


.. image:: https://github.com/JohnGiorgi/seq2rel/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/JohnGiorgi/seq2rel/actions/workflows/ci.yml
   :alt: ci


.. image:: https://codecov.io/gh/JohnGiorgi/seq2rel/branch/main/graph/badge.svg?token=RKJ7EV4WQK
   :target: https://codecov.io/gh/JohnGiorgi/seq2rel
   :alt: codecov


.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://mypy-lang.org/
   :alt: Checked with mypy


.. image:: https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue
   :target: https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue
   :alt: GitHub


A Python package that makes it easy to use sequence-to-sequence (seq2seq) learning for information extraction.

Installation
------------

This repository requires Python 3.7.1 or later. The preferred way to install is via pip:

.. code-block::

   pip install seq2rel

If you need pointers on setting up an appropriate Python environment, please see the `AllenNLP install instructions <https://github.com/allenai/allennlp#installing-via-pip>`_.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^

You can also install from source. 

Using ``pip``\ :

.. code-block::

   pip install git+https://github.com/JohnGiorgi/seq2rel

Using `Poetry <https://python-poetry.org/>`_\ :

.. code-block:: bash

   # Install poetry for your system: https://python-poetry.org/docs/#installation
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

   # Clone and move into the repo
   git clone https://github.com/JohnGiorgi/seq2rel
   cd seq2rel

   # Install the package with poetry
   poetry install

Usage
-----

Preparing a dataset
^^^^^^^^^^^^^^^^^^^

Coming soon.

Training
^^^^^^^^

Coming Soon.

Hyperparameter tuning
~~~~~~~~~~~~~~~~~~~~~

Coming soon.

Inference
^^^^^^^^^

To use the model as a library, import ``Seq2Rel`` and pass it some text (it accepts both strings and lists of strings)

.. code-block:: python

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
   >>> ['@ADE@ ciprofloxacin @DRUG@ renal insufficiency @EFFECT@ @EOR@']

See the list of available ``PRETRAINED_MODELS`` in `seq2rel/seq2rel.py <seq2rel/seq2rel.py>`_

.. code-block:: bash

   python -c "from seq2rel import PRETRAINED_MODELS ; print(list(PRETRAINED_MODELS.keys()))"
