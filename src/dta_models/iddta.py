import datetime
import time

import numpy as np
import pandas as pd


class IDDTA:
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
        self.train_chemicals = []
        self.train_proteins = []

    def __train_one_hot_encoding(self, train):
        chemicals = pd.get_dummies(train['ligand_id'])
        chemicals['cold'] = np.zeros(len(train), int)  # add extra column for novel molecules
        self.train_chemicals = chemicals.columns
        chemicals = chemicals.values

        proteins = pd.get_dummies(train['prot_id'])
        proteins['cold'] = np.zeros(len(train), int)  # add extra column for novel proteins
        self.train_proteins = proteins.columns
        proteins = proteins.values

        return np.hstack([chemicals, proteins])

    def __test_one_hot_encoding(self, test):
        column_dict_chem = {k: np.zeros(len(test), int) for k in self.train_chemicals}
        column_dict_prot = {k: np.zeros(len(test), int) for k in self.train_proteins}

        test_ligand_map = {k: (k if k in set(self.train_chemicals) else 'cold') for k in test['ligand_id'].unique()}
        test_prot_map = {k: (k if k in set(self.train_proteins) else 'cold') for k in test['prot_id'].unique()}

        df_c = pd.DataFrame(column_dict_chem)
        df_c['ligand_id'] = list(test['ligand_id'])
        df_c['ligand_id'] = df_c['ligand_id'].map(test_ligand_map)
        for chem in df_c['ligand_id'].unique():
            df_c.loc[df_c['ligand_id'] == chem, chem] = 1

        df_p = pd.DataFrame(column_dict_prot)
        df_p['prot_id'] = list(test['prot_id'])
        df_p['prot_id'] = df_p['prot_id'].map(test_prot_map)
        for prot in df_p['prot_id'].unique():
            df_p.loc[df_p['prot_id'] == prot, prot] = 1

        chemicals = df_c.drop(columns='ligand_id').values
        proteins = df_p.drop(columns='prot_id').values

        return np.hstack([chemicals, proteins])

    def train(self, train):
        X_train = self.__train_one_hot_encoding(train)

        start = time.time()
        print('Started weak model training on one hot id vectors')
        self.prediction_model.fit(X_train, train['affinity_score'])
        end = time.time()
        print('Weak model training took:', datetime.timedelta(seconds=end - start))

    def predict(self, test):
        X_test = self.__test_one_hot_encoding(test)
        return self.prediction_model.predict(X_test)
