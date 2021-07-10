import datetime
import time

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

from src.utils import encode_smiles


class BoWDTA:
    def __init__(self, prediction_model, bpe_tokenizer, smi_encoding_vocab_path, strong_LM=False):
        self.bpe_tokenizer = bpe_tokenizer
        self.smi_encoding_vocab_path = smi_encoding_vocab_path
        self.prediction_model = prediction_model
        self.chem_bow_vectorizer = Tokenizer(filters=None, lower=False, oov_token='C')
        self.prot_bow_vectorizer = Tokenizer(filters=None, lower=False, oov_token='$')
        self.strong_LM = strong_LM

    def __preprocess_data_for_bow(self, data):
        data = data.copy()
        if not self.strong_LM:
            data['smiles'] = data['smiles'].apply(encode_smiles, encoding_vocab_path=self.smi_encoding_vocab_path)
        chemicals = self.bpe_tokenizer.chem_tokenizer.identify_words(data['smiles'], out_type='int')
        proteins = self.bpe_tokenizer.prot_tokenizer.identify_words(data['aa_sequence'], out_type='int')
        return chemicals, proteins

    def __get_bow_representations(self, chemicals, proteins):
        X_chem = self.chem_bow_vectorizer.texts_to_matrix(chemicals, mode='freq')
        X_prot = self.prot_bow_vectorizer.texts_to_matrix(proteins, mode='freq')
        return np.hstack([X_chem, X_prot])

    def train(self, train):
        chemicals, proteins = self.__preprocess_data_for_bow(train)
        self.chem_bow_vectorizer.fit_on_texts(chemicals)
        self.prot_bow_vectorizer.fit_on_texts(proteins)

        X_train = self.__get_bow_representations(chemicals, proteins)
        start = time.time()
        print('Started training decision-tree on bow vectors')
        self.prediction_model.fit(X_train, train['affinity_score'])
        end = time.time()
        print('Weak model training took:', datetime.timedelta(seconds=end - start))

    def predict(self, test):
        chemicals, proteins = self.__preprocess_data_for_bow(test)
        X_test = self.__get_bow_representations(chemicals, proteins)
        return self.prediction_model.predict(X_test)
