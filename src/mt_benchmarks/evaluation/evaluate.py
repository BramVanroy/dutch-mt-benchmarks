from mt_benchmarks.cli_parser import CliArgumentParser
from mt_benchmarks.evaluation.evaluate_config import BleuArguments, ChrfArguments, TerArguments


def main():
    parser = CliArgumentParser((BleuArguments, ChrfArguments, TerArguments))

    parsed_args = parser.parse()
    # Explicitly setting datatypes for easier auto-completion in IDEs
    bleu_args: BleuArguments = parsed_args[0]
    chrf_args: ChrfArguments = parsed_args[1]
    ter_args: TerArguments = parsed_args[2]


if __name__ == "__main__":
    main()
