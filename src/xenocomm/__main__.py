import argparse

import anndata as ad
import scanpy as sc

from xenocomm.model import XenocommModel


def read_adata(path: str) -> ad.AnnData:
    if path.endswith(".zarr"):
        return ad.read_zarr(path)
    return sc.read_h5ad(path)


def main():
    parser = argparse.ArgumentParser(
        description="Train xenocomm variational inference model"
    )
    parser.add_argument(
        "adata_mouse", type=str, help="Path to mouse AnnData (.h5ad or .zarr)"
    )
    parser.add_argument(
        "adata_human", type=str, help="Path to human AnnData (.h5ad or .zarr)"
    )
    parser.add_argument(
        "save_path", type=str, help="Output path for trained model (.npz)"
    )
    parser.add_argument("--cell-type", type=str, default=None)
    parser.add_argument("--mix-rate", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=-10)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing saved model",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="rich",
        choices=["rich", "print", "none"],
        help="Output mode: rich (interactive), print (line-by-line), none",
    )

    args = parser.parse_args()

    adata_mouse = read_adata(args.adata_mouse)
    adata_human = read_adata(args.adata_human)

    if args.cell_type is not None:
        adata_mouse = adata_mouse[adata_mouse.obs["cell_type"] == args.cell_type].copy()

    if args.resume:
        model = XenocommModel.load(args.save_path, adata_mouse, adata_human)
    else:
        model = XenocommModel(
            adata_mouse,
            adata_human,
            mix_rate=args.mix_rate,
            cutoff=args.cutoff,
        )
        model.build_model()

    verbose_val = args.verbose if args.verbose != "none" else False
    model.train(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        verbose=verbose_val,
    )

    model.save(args.save_path)


if __name__ == "__main__":
    main()
