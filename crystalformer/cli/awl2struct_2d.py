import argparse

from crystalformer.src.structure import write_structures_from_samples


def build_parser():
    parser = argparse.ArgumentParser(description="Convert CrystalFormer-2D sampled AWXL rows to layer CIF structures")
    parser.add_argument("sample_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--deduplicate", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_csv = write_structures_from_samples(args.sample_csv, args.output_dir, deduplicate=args.deduplicate)
    print(output_csv)


if __name__ == "__main__":
    main()
