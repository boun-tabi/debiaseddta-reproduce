import datetime
import json
import time
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.dta_models import BoWDTA, IDDTA
from src.dta_models import DebiasedDTA, BPEDeepDTA, LMDeepDTA
from src.tokenization_methods import SMIAwareBPE, LanguageModel
from src.utils import init_debiasing_parser


def create_weak_learner(name):
    if name == 'BOWDTA':
        return BoWDTA(DecisionTreeRegressor(), tokenizer, paths['chembl27_vocab'], model_name == "LM")
    if name == 'IDDTA':
        return IDDTA(DecisionTreeRegressor())


parser = init_debiasing_parser()
args = parser.parse_args()
args = vars(args)  # to use as a dict

start = time.time()
mini_val_frac = 0.2

model_name = args['model_name']
weak_learner_name = args['weak_learner_name'].upper()
chem_vocab_size = int(args['chem_vocab_size']) if model_name != 'lmdta' else 'NA'
prot_vocab_size = int(args['prot_vocab_size']) if model_name != 'lmdta' else 'NA'
dataset = args['dataset']
n_bootstrapping = args['n_bootstrapping']
decay_type = args['decay_type']
lm_ligand_path = args['lm_ligand_path']
lm_protein_path = args['lm_protein_path']
standardize_labels = args["scale"]
val = args['val']

with open('./paths.json') as f:
    paths = json.load(f)
    dataset_paths = paths[dataset]

decay_mode = 'BD' if decay_type=='lin_decrase' else 'BG'
model_dir = paths['models'] + f'{dataset}/debiaseddta/{model_name}/{weak_learner_name}-{decay_mode}/'

DEBUG_MODE = False
test_scores = []

for setup_ix in range(5):
    train = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/train.csv')
    test_data = {}
    test_data['warm'] = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/test_warm.csv')
    test_data['cold_lig'] = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/test_cold_lig.csv')
    test_data['cold_prot'] = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/test_cold_prot.csv')
    test_data['cold_both'] = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/test_cold_both.csv')
    if val:
        test_data['val'] = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/val_warm.csv')
    scaler = None
    if standardize_labels:
        scaler = StandardScaler()
        train["affinity_score"] = scaler.fit_transform(train["affinity_score"].values.reshape(-1, 1))
        for set_type in test_data:
            test_data[set_type]["affinity_score"] = scaler.transform(
                test_data[set_type]["affinity_score"].values.reshape(-1, 1))
    
    if model_name == 'bpedta':
        strong_model_params_path = paths['models'] + f'{dataset}/{model_name}/chem_{chem_vocab_size}_prot_{prot_vocab_size}/setup_{setup_ix}/params.json'
    else:
        strong_model_params_path = paths['models'] + f'{dataset}/{model_name}/setup_{setup_ix}/params.json'
    with open(strong_model_params_path) as f:
        strong_model_params = json.load(f)

    if DEBUG_MODE:
        train = train.iloc[:100, :]
        test_data = {k: v.iloc[:100, :] for k, v in test_data.items()}
        strong_model_params['n_epochs'] = 1
        n_bootstrapping = 2

    tokenization_methods = {'bpedta': SMIAwareBPE, 'deepdta':SMIAwareBPE,  'lmdta': LanguageModel}
    strong_learner_methods = {'bpedta': BPEDeepDTA, 'deepdta':BPEDeepDTA, 'lmdta': LMDeepDTA}
    tokenization_method = tokenization_methods[model_name]
    strong_learner_method = strong_learner_methods[model_name]

    extra_tokenizer_args = {'dataset_name': dataset,
                            'lm_protein_path': lm_protein_path,
                            'lm_ligand_path': lm_ligand_path
                            }
    tokenizer = tokenization_method(paths, chem_vocab_size, prot_vocab_size, **extra_tokenizer_args)
    if model_name == 'LM':
        strong_model_params['chem_vocab_size'] = tokenizer.chem_vocab_size
        strong_model_params['prot_vocab_size'] = tokenizer.prot_vocab_size

    strong_model_params["scaler"] = scaler

    strong_learner = strong_learner_method(strong_model_params, tokenizer, paths['chembl27_vocab'])
    weak_learner = create_weak_learner(weak_learner_name)
    debiaseddta = DebiasedDTA(weak_learner, strong_learner, mini_val_frac, n_bootstrapping, decay_type)

    setup_dir = model_dir + f'setup_{setup_ix}/'
    if val:
        debiaseddta.train(train, val_data=test_data['val'], savedir=setup_dir)
    else:
        debiaseddta.train(train, savedir=setup_dir)

    setup_scores = debiaseddta.evaluate(test_data, setup_dir, 'test')
    test_scores.append(setup_scores)
    debiaseddta.save(setup_dir)
    debiaseddta.plot_loss(setup_dir)

cv_test_results = {}
for fold in ['warm', 'cold_lig', 'cold_prot', 'cold_both']:
    cv_test_results[fold] = {}
    for metric in ['mse', 'ci', 'rmse', 'r2']:
        cv_test_results[fold][metric] = {}
        metric_scores = [score[fold][metric] for score in test_scores]
        mean = np.mean(metric_scores)
        std = np.std(metric_scores)
        cv_test_results[fold][metric]['mean'] = mean
        cv_test_results[fold][metric]['std'] = std

with open(model_dir + 'test_scores.json', 'w') as f:
    json.dump(cv_test_results, f, indent=4)

end = time.time()
print('The program took:', datetime.timedelta(seconds=end - start))
