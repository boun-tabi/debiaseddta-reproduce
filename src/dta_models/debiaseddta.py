import os
import pickle

import numpy as np
import pandas as pd


class DebiasedDTA:
    def __init__(self, weak_learner, strong_learner, mini_val_frac, n_bootstrapping, decay_type=None):
        self.weak_learner = weak_learner
        self.strong_learner = strong_learner
        self.mini_val_frac = mini_val_frac
        self.n_bootstrapping = n_bootstrapping
        self.decay_type = decay_type

    def learn_sample_weights(self, train_data, savedir=None):
        train = train_data.copy()
        train['interaction_id'] = list(range(len(train)))
        mini_val_data_size = int(len(train) * self.mini_val_frac) + 1
        interaction_id_to_sq_diff = [[] for i in range(len(train))]

        for i in range(self.n_bootstrapping):
            print(f'Bootstrapping ix:{i + 1}/{self.n_bootstrapping}')
            train = train.sample(frac=1)  # shuffle
            n_mini_val = int(1 / self.mini_val_frac)
            for mini_val_ix in range(n_mini_val):
                print(f'Mini val ix:{mini_val_ix + 1}/{n_mini_val}')
                val_start_ix = mini_val_ix * mini_val_data_size
                val_end_ix = val_start_ix + mini_val_data_size
                mini_val = train.iloc[val_start_ix: val_end_ix, :]
                mini_train = pd.concat([train.iloc[:val_start_ix, :],
                                        train.iloc[val_end_ix:, :]])
                assert len(mini_train) + len(mini_val) == len(train)

                self.weak_learner.train(mini_train)
                preds = self.weak_learner.predict(mini_val)
                mini_val['sq_diff'] = (mini_val['affinity_score'] - preds) ** 2
                dct = mini_val.groupby('interaction_id')['sq_diff'].first().to_dict()
                for k, v in dct.items():
                    interaction_id_to_sq_diff[k].append(v)

        for ix, l in enumerate(interaction_id_to_sq_diff):
            assert len(l) == self.n_bootstrapping

        interaction_id_to_med_diff = [np.median(diffs) for diffs in interaction_id_to_sq_diff]
        weights = [med / sum(interaction_id_to_med_diff) for med in interaction_id_to_med_diff]
        if savedir is not None:
            train['sq_diff'] = interaction_id_to_med_diff
            train['weights'] = weights
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            train.to_csv(savedir + 'train_weights.csv', index=None)
            with open(savedir + "weak_model.pkl", "wb") as f:
                pickle.dump(self.weak_learner, f)
        return np.array(weights)

    def train(self, train_data, val_data=None, savedir=None):
        sample_weights = self.learn_sample_weights(train_data, savedir)
        return self.strong_learner.train(train_data, val_data, sample_weights, self.decay_type)

    def only_weak_train(self, train_data, savedir=None):
        return self.learn_sample_weights(train_data, savedir, sample_path="")

    def evaluate(self, test_data, savedir=None, mode='train'):
        return self.strong_learner.evaluate(test_data, savedir, mode)

    def save(self, savedir):
        self.strong_learner.save(savedir)

    def plot_loss(self, savedir):
        self.strong_learner.plot_loss(savedir)
