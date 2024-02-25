from dataclasses import dataclass, field

from sacrebleu.metrics.bleu import MAX_NGRAM_ORDER
from sacrebleu.metrics.chrf import CHRF


@dataclass
class BleuArguments:
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
            "help": "The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default."
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
        metadata={
            "help": f"If given, it overrides the maximum n-gram order (default: {MAX_NGRAM_ORDER}) when computing precisions."
        },
    )
    bleu_effective_order: bool = field(
        default=False,
        metadata={
            "help": "If `True`, stop including n-gram orders for which precision is 0. This should be `True`, if sentence-level BLEU will be computed."
        },
    )
    bleu_trg_lang: str = field(
        default="",
        metadata={"help": "An optional language code to raise potential tokenizer warnings."},
    )


@dataclass
class ChrfArguments:
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
class TerArguments:
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
class CometArguments:
    pass


@dataclass
class BleurtArguments:
    pass


@dataclass
class BertscoreArguments:
    pass
