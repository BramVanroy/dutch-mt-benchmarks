from os import PathLike
from pathlib import Path
from typing import Literal

from huggingface_hub import repo_info


def get_revision(
    repo_id: str,
    *,
    repo_type: Literal["model", "dataset"] = "model",
    revision: str | None = None,
    token: bool | str | None = None,
) -> str:
    """
    Get the revision hash of a dataset or model in the local cache. If a revision is not given, get
    the most recent one.

    :param repo_id: The name of the dataset or model.
    :param repo_type: Repository type to get information from. 'model' or 'dataset'.
    :param revision: The revision hash to get. If None, get the most recent one. 'main' is not a valid revision
    because every 'new' revision is 'main' until it is changed. So if 'main' is passed, it will be replaced by None.
    :param token: The Hugging Face API token to use for authentication. This is useful if you want to access
    private datasets or models. If None, the token will be taken from the currently logged in HF user.
    :return: The revision hash of the item
    """
    if repo_type not in ["model", "dataset"]:
        raise ValueError("repo_type must be either 'model' or 'dataset'. Getting space revision is not supported.")

    return repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    ).sha


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
