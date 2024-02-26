from os import PathLike
from pathlib import Path
from typing import Literal

from huggingface_hub import dataset_info, model_info


def get_revision(
    item_name: str,
    item_type: Literal["dataset", "model"],
    revision: str | None = None,
    token: bool | str | None = None,
) -> str:
    """
    Get the revision hash of a dataset or model in the local cache. If a revision is not given, get
    the most recent one.

    :param item_name: The name of the dataset or model.
    :param item_type: The type of the item, either "dataset" or "model".
    :param revision: The revision hash to get. If None, get the most recent one. 'main' is not a valid revision
    because every 'new' revision is 'main' until it is changed. So if 'main' is passed, it will be replaced by None.
    :param token: The Hugging Face API token to use for authentication. This is useful if you want to access
    private datasets or models. If None, the token will be taken from the currently logged in HF user.
    :return: The revision hash of the item
    """
    revision = None if revision == "main" else revision

    if item_type == "model":
        return model_info(item_name, revision=revision, token=token).sha
    elif item_type == "dataset":
        return dataset_info(item_name, revision=revision, token=token).sha
    else:
        raise ValueError("item_type must be either 'model' or 'dataset'.")


def get_new_result_version(output_dir: str | PathLike) -> int:
    """
    Get the version number for the output directory. If the directory does not exist or f there are no subdirectories,
    return 1. Otherwise, return the latest version number + 1.

    :param output_dir: The directory to get the version number for.
    :return: The version number for the output directory.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        return 1

    subdirs = [subdir for subdir in output_dir.iterdir() if subdir.is_dir()]
    if not subdirs:
        return 1

    latest_version = sorted([int(subdir.name[1:]) for subdir in subdirs], reverse=True)[0]
    return latest_version + 1
