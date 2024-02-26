import json
import logging
import platform
import sys
from pathlib import Path

import datasets
import mt_benchmarks
import numpy as np
import torch
import transformers
from datasets import load_dataset
from mt_benchmarks.cli_parser import CliArgumentParser, serialize_dataclasses_to_yaml
from mt_benchmarks.translation.translate_config import DataArguments, GenerationArguments, ModelArguments
from mt_benchmarks.translation.utils import get_new_result_version, get_revision
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import set_seed


MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer,
    NllbTokenizer,
    NllbTokenizerFast,
]

logger = logging.getLogger(__name__)


def main():
    # This script was modified from
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
    parser = CliArgumentParser((ModelArguments, DataArguments, GenerationArguments, Seq2SeqTrainingArguments))

    parsed_args = parser.parse()
    # Explicitly setting datatypes for easier auto-completion in IDEs
    model_args: ModelArguments = parsed_args[0]
    data_args: DataArguments = parsed_args[1]
    gen_args: GenerationArguments = parsed_args[2]
    training_args: Seq2SeqTrainingArguments = parsed_args[3]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    version_run = get_new_result_version(training_args.output_dir)
    training_args.output_dir = str(Path(training_args.output_dir) / f"v{version_run}")
    training_args.run_name = training_args.output_dir

    logger.info(f"Set output directory to its version number (v{version_run}): {training_args.output_dir}")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'},"
        f" 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Generation parameters {gen_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    if (
        gen_args.source_prefix is None
        and "t5" in model_args.model_name_or_path
        and "google" in model_args.model_name_or_path
    ):
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    if not training_args.predict_with_generate:
        training_args.predict_with_generate = True
        logger.warning(
            "predict_with_generate is set to False, but it is required to generate the predictions."
            " So we're force-setting predict_with_generate to True."
        )

    if not training_args.do_predict:
        training_args.do_predict = True
        logger.warning(
            "do_predict is set to False, but it is required to generate the predictions."
            " So we're force-setting do_predict to True."
        )

    if training_args.do_train or training_args.do_eval:
        training_args.do_train = False
        training_args.do_eval = False
        logger.warning(
            "do_train or do_eval is set to True, but it is not required to train or evaluate the model."
            " So we're force-setting do_train and do_eval to False."
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        token=model_args.token,
        trust_remote_code=data_args.data_trust_remote_code,
        revision=data_args.dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang_token]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.tgt_lang_token)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined by correctly setting 'tgt_lang_token'"
            " to the target language token (for MBart) or the target language."
        )

    prefix = gen_args.source_prefix if gen_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_predict:
        column_names = raw_datasets[data_args.dataset_split_name].column_names
    else:
        raise ValueError("You must enable 'do_predict' to test the model.")

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        if data_args.tgt_lang_token is None or data_args.src_lang_token is None:
            raise ValueError(
                f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --src_lang_token and "
                "--tgt_lang_token arguments to be set to a language token from its vocabulary."
            )

        tokenizer.src_lang = data_args.src_lang_token
        tokenizer.tgt_lang = data_args.tgt_lang_token

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[gen_args.forced_bos_token] if gen_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Temporarily set max_target_length for training.
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        if "translation" in examples:
            inputs = [ex[data_args.src_column] for ex in examples["translation"]]
            targets = [ex[data_args.tgt_column] for ex in examples["translation"]]
        elif data_args.src_column in examples and data_args.tgt_column in examples:
            inputs = examples[data_args.src_column]
            targets = examples[data_args.tgt_column]
        else:
            raise ValueError(
                "You need to pass a dataset with a 'translation' column which then should be a dictionary with 'src_column' and 'tgt_column"
                " keys, or the dataset itself should have the src_column and tgt_column columns"
            )
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if data_args.dataset_split_name not in raw_datasets:
        raise ValueError(f"Split {data_args.dataset_split_name} not found in the dataset")
    predict_dataset = raw_datasets[data_args.dataset_split_name]
    if data_args.max_test_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_test_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))
    with training_args.main_process_first(desc="test set map pre-processing"):
        try:
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test set",
            )
        except Exception as exc:
            raise ValueError(
                "An error occurred when preparing data. This is likely caused because you did not correctly set"
                " the 'tgt_lang_token' and 'src_lang_token' arguments for the multilingual tokenizer you are using."
            ) from exc
    # Data collator
    label_pad_token_id = -100
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    """
    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        inputs = eval_preds.inputs
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        result = {"bleu": bleu_result, "chrf": chrf_result}

        if inputs is not None:
            inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            # Unpack labels: COMET does not use multiref
            comet_lbls = [label[0] for label in decoded_labels]
            comet_result = comet_metric.compute(
                sources=decoded_inputs,
                predictions=decoded_preds,
                references=comet_lbls,
                gpus=torch.cuda.device_count(),
            )["mean_score"]
            result["comet"] = comet_result

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    """

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluation
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = gen_args.num_beams if gen_args.num_beams is not None else training_args.generation_num_beams

    logger.info("*** Predict ***")

    predict_results = trainer.predict(
        predict_dataset, metric_key_prefix="test", max_length=max_length, num_beams=num_beams
    )

    if trainer.is_world_process_zero():
        predictions = predict_results.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions = [pred.strip() for pred in predictions]

        sources = predict_dataset[data_args.src_column]
        references = predict_dataset[data_args.tgt_column]

        pf_preds = Path(training_args.output_dir) / "translations.jsonl"
        with pf_preds.open("w", encoding="utf-8") as fh_preds:
            for idx, (source, reference, prediction) in enumerate(zip(sources, references, predictions)):
                data = {
                    "index": idx,
                    "source": source,
                    "reference": reference,
                    "prediction": prediction,
                }
                fh_preds.write(json.dumps(data) + "\n")

        # Save translation info
        info = {
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.version.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "mt_benchmarks_version": mt_benchmarks.__version__,
            "model_revision": get_revision(model_args.model_name_or_path, "model", revision=model_args.model_revision),
            "dataset_revision": get_revision(data_args.dataset_name, "dataset", revision=data_args.dataset_revision),
        }
        Path(training_args.output_dir).joinpath("translate_info.json").write_text(json.dumps(info, indent=4))

        # Save arguments like this instead of copying input YAML because the YAML can be overridden
        # with CLI arguments
        Path(training_args.output_dir).joinpath("config.yaml").write_text(serialize_dataclasses_to_yaml(parsed_args))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
