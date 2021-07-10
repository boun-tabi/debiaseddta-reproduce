from argparse import ArgumentParser
import json
import re
from itertools import product


def init_debiasing_parser(parser_type="train"):
    parser = ArgumentParser()
    if parser_type == "test":
        parser.add_argument('--model_name', type=str, required=True, choices=['bpedta', 'deepdta', 'lmdta'])
        parser.add_argument('--model_folder', type=str, required=True)
        parser.add_argument('--test_dataset', type=str, required=True, choices=['bdb', 'kiba'])
        parser.add_argument('--train_dataset', type=str, required=True, choices=['bdb', 'kiba'])
        parser.add_argument('--lm_ligand_path', type=str, required=False,
                            help="This should be given according to the test dataset and the model type!!!")
        parser.add_argument('--lm_protein_path', type=str, required=False,
                            help="This should be given according to the test dataset and the model type!!!")
        parser.add_argument('--chem_vocab_size', type=str, default='NA',
                            help="This should be given according to the test dataset and the model type!!!")
        parser.add_argument('--prot_vocab_size', type=str, default='NA',
                            help="This should be given according to the test dataset and the model type!!!")
        parser.add_argument('--scale', action="store_true", required=False)
        parser.add_argument('--full_data', action='store_true', required=False,
                            help="If it is given, the test set is made the whole data set")

    elif parser_type == "train":
        parser.add_argument('--model_name', type=str, required=True, choices=['bpedta', 'lmdta', 'deepdta'])
        parser.add_argument('--weak_learner_name', type=str, required=True, choices=['bowdta', 'iddta'])
        parser.add_argument('--chem_vocab_size', type=str, default='NA')
        parser.add_argument('--prot_vocab_size', type=str, default='NA')
        parser.add_argument('--dataset', type=str, required=True, choices=['bdb', 'kiba'])
        parser.add_argument('--n_bootstrapping', type=int, required=True)
        parser.add_argument('--decay_type', type=str, required=False, choices=['lin_decrease', 'lin_increase'])
        parser.add_argument('--lm_ligand_path', type=str, required=False,
                            help='it is required when the strong model is LMDTA, it specifies the name of the chemical language model embedding file.')
        parser.add_argument('--lm_protein_path', type=str, required=False,
                            help='it is required when the strong model is LMDTA, it specifies the name of the protein language model embedding file.')
        parser.add_argument('--scale', action="store_true", required=False)
        parser.add_argument('--val', action="store_true", required=False)
    return parser


def dct_product(d):
    keys = d.keys()
    lst = []
    for element in product(*d.values()):
        lst.append(dict(zip(keys, element)))
    return lst


def smiles_segmenter(smi):
    pattern = '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print(smi)
    assert smi == ''.join(tokens)
    return tokens


def encode_smiles(smiles, encoding_vocab_path):
    segments = smiles_segmenter(smiles)
    with open(encoding_vocab_path) as f:
        encoding_vocab = json.load(f)

    return ''.join([encoding_vocab.get(segment, encoding_vocab['[OOV]']) for segment in segments])


def decode_smiles(smiles, vocab_path):
    with open(vocab_path + '.inv') as f:
        decoding_vocab = json.load(f)

    return ''.join([decoding_vocab[char] for char in smiles])
