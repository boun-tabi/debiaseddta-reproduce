import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.dta_models import DeepDTA
from src.utils import dct_product


class HyperParamSearcher:
    def __init__(self, paths, n_cv, n_phase, tokenizer, scale_affinity_scores, dataset_name, decision_fold_names=['warm']):
        self.paths = paths
        self.n_cv = n_cv
        self.n_phase = n_phase
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.decision_fold_names = decision_fold_names
        self.scale_affinity_scores = scale_affinity_scores
        cvs = tokenizer.chem_vocab_size
        pvs = tokenizer.prot_vocab_size
        self.savedir = f'{paths["models"]}{dataset_name}/{tokenizer.name}/'
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def __create_cv_score_header(self, param_names):
        train_header = [f'train_{ix}' for ix in range(self.n_cv)]
        val_header = [f'val_{ix}' for ix in range(self.n_cv)]
        stats_header = ['train_mean', 'train_std', 'val_mean', 'val_std']
        return param_names + train_header + val_header + stats_header

    def search(self, fixed_params, search_space, phase_ix):
        dataset_paths = self.paths[self.dataset_name]
        best_score = 9999
        best_models, best_params = [], {}
        param_names = list(search_space.keys())
        cv_scores = [self.__create_cv_score_header(param_names)]
        search_params = dct_product(search_space)
        scaler = None
        for param_ix, params in enumerate(search_params):
            print(f'Doing {param_ix + 1}/{len(search_params)} params: {params}')
            param_scores, param_models = list(params.values()), []
            train_scores, val_scores = [], []
            tf.keras.backend.clear_session()
            build_params = {**params, **fixed_params}
            for setup_ix in range(self.n_cv):
                train = pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/train.csv')
                decision_folds = [pd.read_csv(dataset_paths['folds'] + f'setup_{setup_ix}/val_{fold}.csv') for fold in
                                  self.decision_fold_names]
                decision_fold = pd.concat(decision_folds)

                if self.scale_affinity_scores:
                    scaler = StandardScaler()
                    train["affinity_score"] = scaler.fit_transform(train["affinity_score"].values.reshape(-1, 1))
                    decision_fold["affinity_score"] = scaler.transform(
                        decision_fold["affinity_score"].values.reshape(-1, 1))
                deepdta = DeepDTA(chem_vocab_size=self.tokenizer.chem_vocab_size,
                                  prot_vocab_size=self.tokenizer.prot_vocab_size,
                                  scaler=scaler,
                                  **build_params)

                pp_train = self.tokenizer.fn_pp(data=train,
                                                max_smi_len=fixed_params['max_smi_len'],
                                                max_prot_len=fixed_params['max_prot_len'],
                                                smi_encoding_vocab_path=self.paths[
                                                    'chembl27_vocab'])  # used only by smiles-token-aware models

                pp_decision_fold = self.tokenizer.fn_pp(data=decision_fold,
                                                        max_smi_len=fixed_params['max_smi_len'],
                                                        max_prot_len=fixed_params['max_prot_len'],
                                                        smi_encoding_vocab_path=self.paths['chembl27_vocab'])

                history = deepdta.train(train_data=pp_train, val_data=pp_decision_fold)

                train_score = history['mean_squared_error'][-1]
                val_score = history['val_mean_squared_error'][-1]
                train_scores.append(train_score)
                val_scores.append(val_score)
                param_models.append(deepdta)

            train_score = np.mean(train_scores)
            train_std = np.std(train_scores)

            val_score = np.mean(val_scores)
            val_std = np.std(val_scores)

            param_scores.extend(train_scores)
            param_scores.extend(val_scores)

            param_scores.append(train_score)
            param_scores.append(train_std)
            param_scores.append(val_score)
            param_scores.append(val_std)

            cv_scores.append(param_scores)
            if val_score < best_score:
                best_score = val_score
                best_params = build_params
                best_models = param_models[:]

        with open(self.savedir + 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        df_cv_scores = pd.DataFrame(cv_scores)
        df_cv_scores.to_csv(self.savedir + f'cv_scores_{phase_ix}.csv', index=False, header=False)
        if phase_ix == self.n_phase:
            self.test_model(best_models, dataset_paths, fixed_params['max_smi_len'], fixed_params['max_prot_len'],
                            scaler=scaler)

        return best_params

    def test_model(self, best_models, dataset_paths, max_smi_len, max_prot_len, scaler=None):
        model_scores = []
        for model_ix, model in enumerate(best_models):
            model.save(self.savedir + f'setup_{model_ix}/')
            test_data = {}
            test_data['warm'] = pd.read_csv(dataset_paths['folds'] + f'setup_{model_ix}/test_warm.csv')
            test_data['cold_lig'] = pd.read_csv(dataset_paths['folds'] + f'setup_{model_ix}/test_cold_lig.csv')
            test_data['cold_prot'] = pd.read_csv(dataset_paths['folds'] + f'setup_{model_ix}/test_cold_prot.csv')
            test_data['cold_both'] = pd.read_csv(dataset_paths['folds'] + f'setup_{model_ix}/test_cold_both.csv')
            if scaler is not None:
                for set_type in test_data:
                    test_data[set_type]["affinity_score"] = scaler.transform(
                        test_data[set_type]["affinity_score"].values.reshape(-1, 1))

            test_data = {name: self.tokenizer.fn_pp(data=fold,
                                                    max_smi_len=max_smi_len,
                                                    max_prot_len=max_prot_len,
                                                    smi_encoding_vocab_path=self.paths['chembl27_vocab'])
                         for name, fold in test_data.items()}

            scores = model.evaluate(test_data, savedir=self.savedir + f'setup_{model_ix}/', mode='test')
            model_scores.append(scores)
            model.plot_loss(self.savedir + f'setup_{model_ix}/')
            print('Test scores:', scores)

        cv_test_results = {}
        for fold in ['warm', 'cold_lig', 'cold_prot', 'cold_both']:
            cv_test_results[fold] = {}
            for metric in ['mse', 'ci', 'rmse', 'r2']:
                cv_test_results[fold][metric] = {}
                metric_scores = [score[fold][metric] for score in model_scores]
                mean = np.mean(metric_scores)
                std = np.std(metric_scores)
                cv_test_results[fold][metric]['mean'] = mean
                cv_test_results[fold][metric]['std'] = std

        with open(self.savedir + 'test_scores.json', 'w') as f:
            json.dump(cv_test_results, f, indent=4)
