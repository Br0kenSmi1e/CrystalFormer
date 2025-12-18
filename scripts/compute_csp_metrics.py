import os
import json
import jax
jax.config.update("jax_enable_x64", True)
import pandas as pd
from jax.flatten_util import ravel_pytree

from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from crystalformer.src.formula import formula_string_to_composition_vector
from crystalformer.src.transformer import make_transformer
import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.sample import make_sample_crystal
from crystalformer.reinforce.reward import get_atoms_from_GLXYZAW


def main(args):
    print("========== Load test structures ==========")
    # selected formulas for evaluation from mp20 testset (see DiffCSP paper for details)
    formula_list = ["Co2Sb2", "Sr2O4", "AlAg4", "YMg3", "Cr4Si4",
                    "Sn4Pd4", "Ag6O2", "Co4B2", "Ba2Cd6", "Bi2F8",
                    "KZnF3", "Cr3CuO8", "Bi4S4Cl4", "Si2(CN2)4", "Hg2S2O8"
                   ]
    formula_list = [Composition(f).reduced_composition.formula for f in formula_list]
    
    test_data = pd.read_csv(args.test_path)
    comp_list = [Composition(test_data.iloc[i]['pretty_formula']).reduced_composition for i in range(len(test_data))]
    test_data['comp'] = comp_list

    test_structures = []
    for formula in formula_list:
        comp = Composition(formula).reduced_composition
        subdata = test_data[test_data['comp']==comp]
        if len(subdata)==0:
            print("No test structure found for formula:", formula)
            continue
        elif len(subdata)>1:
            print("Multiple test structures found for formula:", formula, ". Use the first one.")
        else:
            pass
        struct = Structure.from_str(subdata.iloc[0]['cif'], fmt='cif')
        test_structures.append(struct)
    
    print(len(test_structures), "test structures loaded.")

    print("\n========== Build model ==========")
    key = jax.random.PRNGKey(42)
    params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                      args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, args.embed_size, 
                                      args.atom_types, args.wyck_types,
                                      args.dropout_rate, args.attn_dropout)

    print ("# of transformer params", ravel_pytree(params)[0].size) 

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    print("\n========== Start evaluation ==========")
    w_mask = None  # no mask during evaluation
    sample_crystal = make_sample_crystal(transformer, args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, w_mask, args.top_p, args.temperature, args.K, args.spacegroup)

    ase_adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher()
    is_matched_list = [False]*len(formula_list)
    for idx, formula in enumerate(formula_list):
        print("\n===== Sampling crystals for formula: %s =====" %formula)
        composition = formula_string_to_composition_vector(formula)
        print("Composition vector:", composition)
        key, subkey = jax.random.split(key)
        G, XYZ, A, W, M, L = sample_crystal(subkey, params, args.batchsize, composition)
        atoms_list = [get_atoms_from_GLXYZAW(G[i], L[i], XYZ[i], A[i], W[i]) for i in range(args.batchsize)]
        structures = [ase_adaptor.get_structure(atoms_list[i]) for i in range(args.batchsize)]
        # count how many compoisitons are matched
        composition = [s.composition.reduced_composition for s in structures]
        # count how many compoisitons match formula
        print(f"{sum([c == Composition(formula).reduced_composition for c in composition])} out of {len(composition)} match the formula")
        print("Sampled compositions:", composition)
        is_matched_list[idx] = any([matcher.fit(test_structures[idx], structures[i]) for i in range(args.batchsize)])
        print("Formula:", formula, "Matched:", is_matched_list[idx])

    print("\n========== Summary ==========")
    print("Formula matched results:", is_matched_list)
    print("Overall matched rate: %.2f%%" %(sum(is_matched_list)/len(is_matched_list)*100))

    # save results as json in the restore_path
    results = {
        "formula_list": formula_list,
        "is_matched_list": is_matched_list,
        "overall_matched_rate": sum(is_matched_list)/len(is_matched_list),
        "batchsize": args.batchsize
    }
    results_filename = os.path.join(args.restore_path, "eval_csp.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to %s" %results_filename)


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--restore_path", default="./experimental/adam_cfg_0.5_bs_100_lr_1e-05_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_4_H_8_k_32_m_256_e_256_drop_0.3_0.3/", help="")
    parser.add_argument("--test_path", default='/home/user_zdcao/private/datafile/crystal_gpt/dataset/mp_20/test.csv')
    parser.add_argument("--batchsize", type=int, default=20, help="batch size")

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    group = parser.add_argument_group('transformer parameters')
    group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
    group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
    group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
    group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the g and w of first atom')
    group.add_argument('--transformer_layers', type=int, default=4, help='The number of layers in transformer')
    group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
    group.add_argument('--key_size', type=int, default=32, help='The key size')
    group.add_argument('--model_size', type=int, default=256, help='The model size')
    group.add_argument('--embed_size', type=int, default=256, help='The enbedding size')
    group.add_argument('--dropout_rate', type=float, default=0.3, help='The dropout rate for MLP')
    group.add_argument('--attn_dropout', type=float, default=0.3, help='The dropout rate for attention')

    group = parser.add_argument_group('sampling parameters')
    group.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
    group.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')
    group.add_argument('--K', type=int, default=30, help='top K number of space groups. 0 means we sample spacegroup')
    group.add_argument('--spacegroup', type=int, default=None, help='the spacegroup number 1-230, given that will overwrites K')
    args = parser.parse_args()

    main(args)
