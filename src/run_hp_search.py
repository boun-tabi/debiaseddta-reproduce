import datetime
import json
import sys
import time

import tensorflow as tf

from src.hyper_param_searcher import HyperParamSearcher
from src.tokenization_methods import SMIAwareBPE, LanguageModel

if __name__ == '__main__':
    start = time.time()
    tf.random.set_seed(1)
    method = sys.argv[1].lower()
    if method not in {'deepdta', 'bpedta', 'lmdta'}:
        raise ValueError('Supported method names are {deepdta, bpedta, lmdta}')
    dataset = sys.argv[2].lower()
    if dataset not in {'bdb', 'kiba'}:
        raise ValueError('Supported dataset names are {bdb, kiba}')

    chem_vocab_sizes = {'deepdta': [94], 'bpedta': [8000, 16000, 32000], 'lmdta': [0]}
    prot_vocab_sizes = {'deepdta': [26], 'bpedta': [8000, 16000, 32000], 'lmdta': [0]}
    hp_space_paths = {'deepdta': 'deepdta_hp_space.json', 'bpedta': 'deepdta_hp_space.json',
                      'lmdta': 'lmdta_hp_space.json'}

    chem_vocab_sizes = chem_vocab_sizes[method]
    prot_vocab_sizes = prot_vocab_sizes[method]
    hp_space_path = hp_space_paths[method]
    scale_affinity_scores = method == 'lmdta'
    tokenization_methods = {'deepdta': SMIAwareBPE, 'bpedta': SMIAwareBPE, 'lmdta': LanguageModel}

    with open('paths.json') as f:
        paths = json.load(f)

    with open(hp_space_path) as f:
        hp_space = json.load(f)

    fixed_params = hp_space['fixed_params']
    n_cv = 5
    for chem_vocab_size in chem_vocab_sizes:
        for prot_vocab_size in prot_vocab_sizes:
            extra_tokenizer_args = {'dataset_name': dataset, 'lm_ligand_path': f'data/{dataset}/ligand_embed.json',
                                    'lm_protein_path': f'data/{dataset}/protein_embed.json'}
            tokenization_method = tokenization_methods[method](paths, chem_vocab_size, prot_vocab_size,
                                                               **extra_tokenizer_args)
            n_phase = len(hp_space['search_params'])
            best_params = {}
            searcher = HyperParamSearcher(paths,
                                          n_cv,
                                          n_phase,
                                          tokenization_method,
                                          scale_affinity_scores,
                                          dataset)
            for phase in range(n_phase):
                fixed_params = {**best_params, **fixed_params}
                search_params = hp_space['search_params'][phase]
                best_params = searcher.search(fixed_params, search_params, phase + 1)

    end = time.time()
    print('The program took:', datetime.timedelta(seconds=end - start))
