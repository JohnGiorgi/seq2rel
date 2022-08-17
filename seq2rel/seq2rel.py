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
    "cdr": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/cdr.tar.gz",
    "cdr_hints": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/cdr_hints.tar.gz",
    "gda": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/gda.tar.gz",
    "gda_hints": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/gda_hints.tar.gz",
    "dgm": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/dgm.tar.gz",
    "dgm_hints": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/dgm_hints.tar.gz",
    "docred": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/docred.tar.gz",
}


class Seq2Rel:
    """A simple interface to the model for the purposes of extracting entities and relations from text.

    # Example Usage

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
    def __call__(self, inputs: Union[str, List[str]], batch_size: Optional[int] = 32) -> List[str]:
        """Returns a list of strings, one for each item in `inputs`.

        # Parameters

        inputs : `Union[str, List[str]]`, required
            The input text to extract relations from. Can be a string, list of strings, or a
            filepath/URL to a text file with one input per line.
        batch_size : `int`, optional, (default = `32`)
            The batch size to use when making predictions.

        # Returns:

        A list of strings, containing the serialized relations extracted from the `inputs`.
        """
        if isinstance(inputs, str):
            try:
                if Path(inputs).is_file() or url(inputs):
                    inputs = Path(cached_path(inputs)).read_text().strip().split("\n")
                else:
                    inputs = [inputs]  # type: ignore
            except OSError:
                inputs = [inputs]  # type: ignore

        predicted_strings = []
        for batch in chunked(inputs, batch_size):
            batch_json = [{"source": sanitize_text(example)} for example in batch]
            outputs = self._predictor.predict_batch_json(batch_json)
            outputs = [output[self._output_dict_field] for output in outputs]
            predicted_strings.extend(outputs)

        return predicted_strings
