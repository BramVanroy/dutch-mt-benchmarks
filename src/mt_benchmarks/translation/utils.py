from os import PathLike
from pathlib import Path
from typing import Literal

from huggingface_hub import scan_cache_dir


def get_revision(item_name: str, item_type: Literal["dataset", "model"], revision: str | None = None) -> str | None:
    """
    Get the revision hash of a dataset or model in the local cache. If a revision is not given, get
    the most recent one.

    :param item_name: The name of the dataset or model.
    :param item_type: The type of the item, either "dataset" or "model".
    :param revision: The revision hash to get. If None, get the most recent one. 'main' is not a valid revision
    because every 'new' revision is 'main' until it is changed. So if 'main' is passed, it will be replaced by None.
    :return: The revision hash of the item, or None if it does not exist - although that should not occur.
    """
    revision = None if revision == "main" else revision
    hf_cache_info = scan_cache_dir()
    for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path):
        if item_type != repo.repo_type or item_name != repo.repo_id:
            continue
        revisions = repo.revisions

        if revision is not None:
            revision = next((r for r in revisions if r.commit_hash == revision), None)
        else:
            # Use the most recent one
            revisions = sorted(revisions, key=lambda revision: revision.last_modified, reverse=True)
            revision = revisions[0]

        return revision.commit_hash if revision is not None else None


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
