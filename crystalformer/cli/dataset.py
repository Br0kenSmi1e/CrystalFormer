import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="CrystalFormer-2D layer CSV preprocessing")
    parser.add_argument("--path", type=str, required=True, help="Directory containing split CSV files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--n_max", type=int, default=21)
    parser.add_argument("--atom_types", type=int, default=119)
    parser.add_argument("--wyck_types", type=int, default=19)
    parser.add_argument("--num_workers", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from crystalformer.src.utils import GLXYZAW_from_file

    root = Path(args.path)

    for split in args.splits:
        csv_file = root / f"{split}.csv"
        if not csv_file.is_file():
            raise SystemExit(f"Missing layer CSV: {csv_file}")
        G, L, XYZ, A, W = GLXYZAW_from_file(
            csv_file,
            atom_types=args.atom_types,
            wyck_types=args.wyck_types,
            n_max=args.n_max,
            num_workers=args.num_workers,
        )
        print(
            f"{split}: G={G.shape} L={L.shape} XYZ={XYZ.shape} "
            f"A={A.shape} W={W.shape}"
        )


if __name__ == "__main__":
    main()
