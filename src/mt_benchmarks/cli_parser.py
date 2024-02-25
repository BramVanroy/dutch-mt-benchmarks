import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType


class CliArgumentParser(HfArgumentParser):
    # Adapted from
    # https://github.com/huggingface/alignment-handbook/blob/87cc800498b17432cfb7f5acb5e9a79f15c867fc/src/alignment/configs.py#L32
    def parse_yaml_and_args(self, yaml_arg: str, other_args: list[str] | None = None) -> list[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args} if other_args else {}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == list[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output

    def parse_yaml_file(self, yaml_file: str, allow_extra_keys: bool = False) -> tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a yaml file and populating the
        dataclass types.

        Args:
            yaml_file (`str` or `os.PathLike`):
                File name of the yaml file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        outputs = self.parse_dict(
            yaml.load(Path(yaml_file).read_text(encoding="utf-8"), Loader=yaml.Loader),
            allow_extra_keys=allow_extra_keys,
        )
        return tuple(outputs)


def serialize_dataclasses_to_yaml(dataclass_instance: list[dataclass]) -> str:
    """
    Serialize a list of dataclasses to a yaml string. Ignores keys starting with "_"
    :param dataclass_instance: list of dataclasses
    :return: yaml string
    """
    attrs = {}
    for dc in dataclass_instance:
        data_dict = asdict(dc)

        for key, value in data_dict.items():
            if key.startswith("_"):
                continue
            try:
                yaml.dump({key: value})
                if key in attrs:
                    raise KeyError(f"Duplicate key {key} found in dataclass instances")
                attrs[key] = value
            except TypeError:
                continue

    # sort keys
    attrs = dict(sorted(attrs.items()))

    return yaml.dump(attrs)
