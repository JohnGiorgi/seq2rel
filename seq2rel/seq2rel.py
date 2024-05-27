import logging
import os
import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.common.meta import Meta
from allennlp.common.params import Params
from allennlp.models import archival
from allennlp.predictors import Predictor
from more_itertools import chunked
from validators.url import url

from seq2rel.common.util import sanitize_text

logger = logging.getLogger(__name__)

PRETRAINED_MODELS = {
    "cdr": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/cdr.tar.gz",
    "cdr_hints": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/cdr_hints.tar.gz",
    "gda": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/gda.tar.gz",
    "gda_hints": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/gda_hints.tar.gz",
    "dgm": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/dgm.tar.gz",
    "dgm_hints": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/dgm_hints.tar.gz",
    "docred": "https://github.com/JohnGiorgi/seq2rel/releases/download/pretrained-models/docred.tar.gz",
}

# Needed strictly to rename
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext with
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# otherwise identical to allennlp.models.archival.load_archive
def load_archive(
    archive_file: Union[str, PathLike],
    cuda_device: int = -1,
    overrides: Union[str, Dict[str, Any]] = "",
    weights_file: str = None,
) -> archival.Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    # Parameters

    archive_file : `Union[str, PathLike]`
        The archive file to load the model from.
    cuda_device : `int`, optional (default = `-1`)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
        JSON overrides to apply to the unarchived `Params` object.
    weights_file : `str`, optional (default = `None`)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    """
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    meta: Optional[Meta] = None

    tempdir = None
    try:
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            with archival.extracted_archive(resolved_archive_file, cleanup=False) as tempdir:
                serialization_dir = tempdir

        if weights_file:
            weights_path = weights_file
        else:
            weights_path = archival.get_weights_path(serialization_dir)

        # Load config
        config = Params.from_file(os.path.join(serialization_dir, archival.CONFIG_NAME), overrides)

        # Rename
        # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext with
        # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
        def rename_pretrained_model(config: Params) -> Params:
            for key, value in config.items():
                if isinstance(value, dict):
                    rename_pretrained_model(value)
                else:
                    if value == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
                        config[
                            key
                        ] = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

        config = config.as_dict()
        rename_pretrained_model(config)
        config = Params(config)

        # Instantiate model and dataset readers. Use a duplicate of the config, as it will get consumed.
        dataset_reader, validation_dataset_reader = archival._load_dataset_readers(
            config.duplicate(), serialization_dir
        )
        model = archival._load_model(
            config.duplicate(), weights_path, serialization_dir, cuda_device
        )

        # Load meta.
        meta_path = os.path.join(serialization_dir, archival.META_NAME)
        if os.path.exists(meta_path):
            meta = Meta.from_path(meta_path)
    finally:
        if tempdir is not None:
            logger.info(f"removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)

    # Check version compatibility.
    if meta is not None:
        archival._check_version_compatibility(archive_file, meta)

    return archival.Archive(
        model=model,
        config=config,
        dataset_reader=dataset_reader,
        validation_dataset_reader=validation_dataset_reader,
        meta=meta,
    )


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
