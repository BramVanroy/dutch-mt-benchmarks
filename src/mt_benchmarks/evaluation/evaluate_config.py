from dataclasses import dataclass, field, fields
from typing import ClassVar

from sacrebleu.metrics.bleu import MAX_NGRAM_ORDER
from sacrebleu.metrics.chrf import CHRF


@dataclass
class MetricBaseArguments:
    name: ClassVar[str]

    def __post_init__(self):
        for fld in fields(self):
            if fld.name != f"use_metric_{self.name}" and not fld.name.startswith(self.name):
                raise ValueError(
                    f"Invalid field name: {fld.name}. Field names must start with 'use_metric_' followed by its name,"
                    f" e.g. 'use_metric_{self.name}' or the metric name, in this case '{self.name}'."
                )


@dataclass
class BleuArguments(MetricBaseArguments):
    name = "bleu"

    use_metric_bleu: bool = field(
        default=True,
        metadata={"help": "Whether to use BLEU for evaluating translations."},
    )
    bleu_lowercase: bool = field(
        default=False,
        metadata={"help": "If True, lowercased BLEU is computed."},
    )
    bleu_force: bool = field(
        default=False,
        metadata={"help": "Ignore data that looks already tokenized."},
    )
    bleu_tokenize: str | None = field(
        default=None,
        metadata={
            "help": "The tokenizer to use. If None, defaults to language-specific tokenizers"
            " with '13a' as the fallback default."
        },
    )
    bleu_smooth_method: str = field(
        default="exp",
        metadata={"help": "The smoothing method to use ('floor', 'add-k', 'exp' or 'none')."},
    )
    bleu_smooth_value: float | None = field(
        default=None,
        metadata={"help": "The smoothing value for `floor` and `add-k` methods. `None` falls back to default value."},
    )
    bleu_max_ngram_order: int = field(
        default=MAX_NGRAM_ORDER,
        metadata={"help": f"If given, it overrides the maximum n-gram order when computing precisions."},
    )
    bleu_effective_order: bool = field(
        default=False,
        metadata={
            "help": "If `True`, stop including n-gram orders for which precision is 0. This should be `True`,"
            " if sentence-level BLEU will be computed."
        },
    )
    bleu_trg_lang: str = field(
        default="",
        metadata={"help": "An optional language code to raise potential tokenizer warnings."},
    )


@dataclass
class ChrfArguments(MetricBaseArguments):
    name = "chrf"

    use_metric_chrf: bool = field(
        default=True,
        metadata={"help": "Whether to use ChrF for evaluating translations."},
    )
    chrf_char_order: int = field(
        default=CHRF.CHAR_ORDER,
        metadata={"help": "Character n-gram order."},
    )
    chrf_word_order: int = field(
        default=CHRF.WORD_ORDER,
        metadata={"help": "Word n-gram order. If equals to 2, the metric is referred to as chrF++."},
    )
    chrf_beta: int = field(
        default=CHRF.BETA,
        metadata={"help": "Determine the importance of recall w.r.t precision."},
    )
    chrf_lowercase: bool = field(
        default=False,
        metadata={"help": "Enable case-insensitivity."},
    )
    chrf_whitespace: bool = field(
        default=False,
        metadata={"help": "If `True`, include whitespaces when extracting character n-grams."},
    )
    chrf_eps_smoothing: bool = field(
        default=False,
        metadata={
            "help": "If `True`, applies epsilon smoothing similar to reference chrF++.py, NLTK and Moses"
            " implementations. Otherwise, it takes into account effective match order similar to"
            " sacreBLEU < 2.0.0."
        },
    )


@dataclass
class TerArguments(MetricBaseArguments):
    name = "ter"

    use_metric_ter: bool = field(
        default=True,
        metadata={"help": "Whether to use TER for evaluating translations."},
    )
    ter_normalized: bool = field(
        default=False,
        metadata={
            "help": "Enable character normalization. By default, normalizes a couple of things such as newlines being"
            " stripped, retrieving XML encoded characters, and fixing tokenization for punctuation. When"
            " 'asian_support' is enabled, also normalizes specific Asian (CJK) character sequences, i.e."
            " split them down to the character level."
        },
    )
    ter_no_punct: bool = field(
        default=False,
        metadata={
            "help": "Remove punctuation. Can be used in conjunction with 'asian_support' to also remove typical "
            "punctuation markers in Asian languages (CJK)."
        },
    )
    ter_asian_support: bool = field(
        default=False,
        metadata={
            "help": "Enable special treatment of Asian characters. This option only has an effect when 'normalized'"
            " and/or 'no_punct' is enabled. If 'normalized' is also enabled, then Asian (CJK)"
            " specific unicode ranges for CJK and full-width punctuations are also removed."
        },
    )
    ter_case_sensitive: bool = field(
        default=False,
        metadata={"help": "If `True`, does not lowercase sentences."},
    )


@dataclass
class CometArguments(MetricBaseArguments):
    name = "comet"

    use_metric_comet: bool = field(
        default=True,
        metadata={"help": "Whether to use COMET for evaluating translations."},
    )
    comet_model_name: str = field(
        default="Unbabel/wmt22-comet-da",
        metadata={"help": "The name of the COMET model to use."},
    )
    comet_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size used during inference."},
    )
    comet_gpus: int = field(
        default=1,
        metadata={"help": "Number of GPUs to use."},
    )
    comet_mc_dropout: int = field(
        default=0,
        metadata={"help": "Number of inference steps to run using Monte Carlo dropout."},
    )
    comet_progress_bar: bool = field(
        default=True,
        metadata={"help": "Flag that turns on and off the predict progress bar."},
    )
    comet_accelerator: str = field(
        default="auto",
        metadata={"help": "Pytorch Lightning accelerator (e.g: 'cpu', 'cuda', 'hpu', 'ipu', 'mps', 'tpu')."},
    )
    comet_num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers to use when loading and preparing data."},
    )
    comet_length_batching: bool = field(
        default=True,
        metadata={"help": "If set to true, reduces padding by sorting samples by sequence length."},
    )


@dataclass
class BleurtArguments(MetricBaseArguments):
    name = "bleurt"

    use_metric_bleurt: bool = field(
        default=False,
        metadata={"help": "Whether to use BLEURT for evaluating translations."},
    )
    bleurt_model_name: str = field(
        default="BLEURT-20",
        metadata={"help": "The name of the BLEURT model to use."},
    )
    bleurt_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size used during inference."},
    )
    # TODO: not sure if we can control which device it uses? Cannot seem to find any
    # flags in the repo. Perhaps it just auto-decides based on available hardware?
    # That would be unfortunate, as it would be nice to be able to control this.


@dataclass
class BertscoreArguments(MetricBaseArguments):
    # IDF is not implemented here because it requires a list of sentences as input, which is not "pretty" to do
    # in the CLI. It is also not a very common use case, so it is better to leave it out for now. If a request is
    # made for it, we can add it but probably by using a file input instead of a CLI argument.
    name = "bertscore"

    use_metric_bertscore: bool = field(
        default=True,
        metadata={"help": "Whether to use BERTScore for evaluating translations."},
    )
    bertscore_model_type: str | None = field(
        default=None,
        metadata={
            "help": "Contexual embedding model specification, default using the suggested model for the target"
            " language; has to specify at least one of `model_type` or `lang`."
        },
    )
    bertscore_lang: str | None = field(
        default=None,
        metadata={
            "help": "language of the sentences. Either 'lang' has to be specified so that a default model can be"
            " used, or a model has to be specified in `model_type`."
        },
    )
    bertscore_num_layers: int | None = field(
        default=None,
        metadata={
            "help": "The layer of representation to use. Defaults to using the number of layer tuned on"
            " WMT16 correlation data for the given model."
        },
    )
    bertscore_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size used during inference."},
    )
    bertscore_nthreads: int = field(
        default=4,
        metadata={"help": "Number of threads used during inference."},
    )
    bertscore_device: str | None = field(
        default=None,
        metadata={
            "help": "Device on which the contextual embedding model will be allocated on. If this argument is"
            " None, the model lives on cuda:0 if cuda is available."
        },
    )
    bertscore_rescale_with_baseline: bool = field(
        default=False,
        metadata={"help": "Whether to rescale BERTScore with pre-computed baseline."},
    )
    bertscore_baseline_path: str | None = field(
        default=None,
        metadata={"help": "Path to the pre-computed baseline for rescaling BERTScore."},
    )
    bertscore_use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use a fast HF tokenizer."},
    )
