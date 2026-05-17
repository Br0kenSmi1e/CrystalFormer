import argparse
import math
import multiprocessing
import os

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
import pandas as pd

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.lattice import norm_lattice
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.sample import make_sample_crystal
from crystalformer.src.train import train
from crystalformer.src.transformer import make_transformer
from crystalformer.src.utils import GLXYZAW_from_file


np.set_printoptions(threshold=np.inf)


def build_parser():
    parser = argparse.ArgumentParser(description="CrystalFormer-2D fixed layer-group generation")
    parser.add_argument("--run_name", type=str, default="crystalformer-2d")
    parser.add_argument("--data", type=str, default="data/mini.csv")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "none"])
    parser.add_argument("--restore_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="ckpt")
    parser.add_argument("--layergroup", type=int, default=None, help="fixed layer group in [1, 80] for sampling")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n_max", type=int, default=21)
    parser.add_argument("--atom_types", type=int, default=119)
    parser.add_argument("--wyck_types", type=int, default=19)
    parser.add_argument("--Kx", type=int, default=16)
    parser.add_argument("--Kl", type=int, default=4)
    parser.add_argument("--Nf", type=int, default=5)
    parser.add_argument("--h0_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--key_size", type=int, default=8)
    parser.add_argument("--model_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def validate_args(args):
    if args.optimizer == "none":
        if args.restore_path is None:
            raise SystemExit("--restore_path is required when --optimizer none")
        if args.layergroup is None:
            raise SystemExit("--layergroup is required when --optimizer none")
        if not 1 <= args.layergroup <= 80:
            raise SystemExit("--layergroup must be in [1, 80]")
    return args


def main(argv=None):
    parser = build_parser()
    args = validate_args(parser.parse_args(argv))

    key = jax.random.PRNGKey(args.seed)
    num_cpu = multiprocessing.cpu_count()
    num_io_process = min(num_cpu, args.batchsize)
    print("number of available cpu: ", num_cpu)

    if args.optimizer == "none":
        try:
            ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path)
        except FileNotFoundError:
            ckpt_filename, epoch_finished = None, 0
        if ckpt_filename is None:
            raise SystemExit(f"No checkpoint found at --restore_path {args.restore_path}")
        try:
            ckpt = checkpoint.load_data(ckpt_filename)
        except Exception as exc:
            raise SystemExit(f"Failed to load checkpoint at --restore_path {args.restore_path}: {exc}") from exc
        if not isinstance(ckpt, dict) or "params" not in ckpt:
            raise SystemExit(f"Checkpoint at --restore_path {args.restore_path} is missing params")
    else:
        ckpt_filename, epoch_finished = None, 0
        train_data = GLXYZAW_from_file(args.data, args.atom_types, args.wyck_types, args.n_max, num_io_process)
        train_groups = np.asarray(train_data[0])
        if np.any((train_groups < 1) | (train_groups > 80)):
            raise SystemExit("CrystalFormer-2D training data must use layergroup values in [1, 80]")
        valid_data = train_data
        ckpt = {}

    params, transformer = make_transformer(
        key,
        args.Nf,
        args.Kx,
        args.Kl,
        args.n_max,
        args.h0_size,
        args.num_layers,
        args.num_heads,
        args.key_size,
        args.model_size,
        args.embed_size,
        args.atom_types,
        args.wyck_types,
        args.dropout_rate,
        args.attn_dropout,
    )
    transformer_name = "Nf_%d_Kx_%d_Kl_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_e_%d_drop_%g_%g" % (
        args.Nf,
        args.Kx,
        args.Kl,
        args.h0_size,
        args.num_layers,
        args.num_heads,
        args.key_size,
        args.model_size,
        args.embed_size,
        args.dropout_rate,
        args.attn_dropout,
    )
    print("# of transformer params", ravel_pytree(params)[0].size)

    loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer)

    print("\n========== Prepare logs ==========")
    if args.optimizer != "none":
        output_path = os.path.join(args.save_path, args.run_name + "_" + transformer_name)
        os.makedirs(output_path, exist_ok=True)
        print("Create directory for output: %s" % output_path)
    else:
        output_path = args.save_path
        os.makedirs(output_path, exist_ok=True)
        print("Will output samples to: %s" % output_path)

    print("\n========== Load checkpoint==========")
    if args.optimizer != "none":
        try:
            ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or output_path)
        except FileNotFoundError:
            ckpt_filename, epoch_finished = None, 0
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" % (ckpt_filename, epoch_finished))
        if args.optimizer != "none":
            ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        ckpt = {}
        print("No checkpoint file found. Start from scratch.")

    if args.optimizer != "none":
        schedule = lambda t: 1e-4
        optimizer = optax.chain(
            optax.clip(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=0.0),
        )

        opt_state = optimizer.init(params)
        if "opt_state" in ckpt:
            opt_state = ckpt["opt_state"]

        print("\n========== Start training ==========")
        def train_loss_fn(params, key, _composition, G, L, XYZ, A, W, is_train):
            loss, (loss_w, loss_a, loss_xyz, loss_l) = loss_fn(params, key, G, L, XYZ, A, W, is_train)
            loss_g = jnp.zeros_like(loss_w)
            return loss, (loss_g, loss_w, loss_a, loss_xyz, loss_l)

        params, opt_state = train(
            key,
            optimizer,
            opt_state,
            train_loss_fn,
            params,
            epoch_finished,
            args.epochs,
            args.batchsize,
            train_data,
            valid_data,
            output_path,
            100,
            0.0,
        )
        return params, opt_state

    print("\n========== Start sampling ==========")
    sampler = make_sample_crystal(
        transformer,
        args.n_max,
        args.atom_types,
        args.wyck_types,
        args.Kx,
        args.Kl,
        w_mask=None,
        top_p=args.top_p,
        temperature=args.temperature,
        layergroup=args.layergroup,
    )

    num_batches = math.ceil(args.num_samples / args.batchsize)
    filename = os.path.join(output_path, "samples.csv")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, args.num_samples)
        n_sample = end_idx - start_idx
        key, subkey = jax.random.split(key)
        G, XYZ, A, W, M, L = sampler(subkey, params, n_sample)

        if n_sample == 0:
            continue

        if args.num_samples <= args.batchsize:
            G, XYZ, A, W, M, L = (x[:n_sample] for x in (G, XYZ, A, W, M, L))

        if batch_idx == 0:
            print("targeting layergroup No.", args.layergroup)

        L_for_logp = norm_lattice(G, W, L)
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(
            params, key, G, L_for_logp, XYZ, A, W, False
        )

        data = pd.DataFrame()
        data["G"] = np.array(G).tolist()
        data["L"] = np.array(L).tolist()
        data["X"] = np.array(XYZ).tolist()
        data["A"] = np.array(A).tolist()
        data["W"] = np.array(W).tolist()
        data["M"] = np.array(M).tolist()
        data["logp_w"] = np.array(logp_w).tolist()
        data["logp_xyz"] = np.array(logp_xyz).tolist()
        data["logp_a"] = np.array(logp_a).tolist()
        data["logp_l"] = np.array(logp_l).tolist()

        write_mode = "w" if batch_idx == 0 else "a"
        write_header = batch_idx == 0
        data.to_csv(filename, mode=write_mode, index=False, header=write_header)
        print("Wrote samples to %s (batch %d/%d)" % (filename, batch_idx + 1, num_batches))


if __name__ == "__main__":
    main()
