from pathlib import Path
from typing import List, Optional, Union

import torch
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from validators.url import url

from seq2rel.common.util import sanitize_text

PRETRAINED_MODELS = {
    "ade": "https://github.com/JohnGiorgi/seq2rel/releases/download/v0.1.0rc1/ade.tar.gz",
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
    >>> ['<ADE> ciprofloxacin <DRUG> renal insufficiency <EFFECT> </ADE>']
    ```

    # Parameters

    pretrained_model_name_or_path : `str`, required
        Path to a serialized AllenNLP archive or a model name from:
        `list(seq2rel.PRETRAINED_MODELS.keys())`
    **kwargs : `Dict`, optional
        Keyword arguments that will be passed to `allennlp.models.archival.load_archive`. This is
        useful, for example, to specify a CUDA device id with `cuda_device`. See:
        https://docs.allennlp.org/master/api/models/archival/#load_archive for more details.
    """

    _output_dict_field = "predicted_strings"

    def __init__(self, pretrained_model_name_or_path: str, **kwargs) -> None:
        if pretrained_model_name_or_path in PRETRAINED_MODELS:
            pretrained_model_name_or_path = PRETRAINED_MODELS[pretrained_model_name_or_path]
        common_util.import_module_and_submodules("seq2rel")
        archive = load_archive(pretrained_model_name_or_path, **kwargs)
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
        if isinstance(inputs, str):
            if Path(inputs).is_file() or url(inputs):
                inputs = Path(cached_path(inputs)).read_text().split("\n")
            else:
                inputs = [inputs]

        if batch_size is None:
            batch_size = len(inputs)

        predicted_strings = []
        for i in range(0, len(inputs), batch_size):
            batch_json = [
                {"source": sanitize_text(input_)} for input_ in inputs[i : i + batch_size]
            ]
            outputs = self._predictor.predict_batch_json(batch_json)
            outputs = [output[self._output_dict_field] for output in outputs]
            predicted_strings.extend(outputs)

        return predicted_strings
