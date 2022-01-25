from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from more_itertools import chunked
from validators.url import url

from seq2rel.common.util import sanitize_text

PRETRAINED_MODELS = {
    "bc5cdr": "https://github.com/JohnGiorgi/seq2rel/releases/download/v0.1.0rc1/bc5cdr.tar.gz",
    "gda": "https://github.com/JohnGiorgi/seq2rel/releases/download/v0.1.0rc1/gda.tar.gz",
    "docred": "https://github.com/JohnGiorgi/seq2rel/releases/download/v0.1.0rc1/docred.tar.gz",
}


class Seq2Rel:
    """A simple interface to the model for the purposes of extracting entities and relations from text.

    # Example Usage

    ```python
    from seq2rel import Seq2Rel

    # Pretrained models stored in GitHub. Downloaded and cached automatically.
    # This model is ~500mb.
    pretrained_model = "ade"

    # Models are loaded via a dead-simple interface.
    seq2rel = Seq2Rel(pretrained_model)

    # Extremely flexible inputs. User can provide...
    # - a string
    # - a list of strings
    # - a text file (local path or URL)
    input_text = "Ciprofloxacin-induced renal insufficiency in cystic fibrosis."

    seq2rel(input_text)
    >>> ['ciprofloxacin @DRUG@ renal insufficiency @EFFECT@ @ADE@']
    ```

    # Parameters

    pretrained_model_name_or_path : `str`, required
        Path to a serialized AllenNLP archive or a model name from:
        `list(seq2rel.PRETRAINED_MODELS.keys())`
    **kwargs : `Any`, optional, (default = `{}`)
        Keyword arguments that will be passed to `allennlp.models.archival.load_archive`. This is
        useful, for example, to specify a CUDA device id with `cuda_device`. See:
        https://docs.allennlp.org/main/api/models/archival/#load_archive for more details.
    """

    _output_dict_field = "predicted_strings"

    def __init__(self, pretrained_model_name_or_path: str, **kwargs: Any) -> None:
        if pretrained_model_name_or_path in PRETRAINED_MODELS:
            pretrained_model_name_or_path = PRETRAINED_MODELS[pretrained_model_name_or_path]
        common_util.import_module_and_submodules("seq2rel")
        # Setup any default overrides here. For example, we don't want to load the pretrained
        # weights from HuggingFace because this model has been fine-tuned.
        overrides = {
            "model.source_embedder.token_embedders.tokens.load_weights": False,
        }
        # Allow user to update these with kwargs.
        if "overrides" in kwargs:
            overrides.update(kwargs.pop("overrides"))
        archive = load_archive(pretrained_model_name_or_path, overrides=overrides, **kwargs)
        self._predictor = Predictor.from_archive(archive, predictor_name="seq2seq")

    @torch.no_grad()
    def __call__(
        self, inputs: Union[str, List[str]], batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Returns a numpy array of embeddings, one for each item in `inputs`.

        # Parameters

        inputs : `Union[str, List[str]]`, required
            The input text to embed. Can be a string, list of strings, or a filepath/URL to a text
            file with one input per line.
        batch_size : `int`, optional
            If given, the `inputs` will be batched before embedding.
        """
        # TODO: This is ugly, clean it up.
        if isinstance(inputs, str):
            try:
                if Path(inputs).is_file() or url(inputs):
                    inputs = Path(cached_path(inputs)).read_text().strip().split("\n")
                else:
                    inputs = [inputs]  # type: ignore
            except OSError:
                inputs = [inputs]  # type: ignore

        if batch_size is None:
            batch_size = len(inputs)

        predicted_strings = []
        for batch in chunked(inputs, batch_size):
            batch_json = [{"source": sanitize_text(example)} for example in batch]
            outputs = self._predictor.predict_batch_json(batch_json)
            outputs = [output[self._output_dict_field] for output in outputs]
            predicted_strings.extend(outputs)

        return predicted_strings
