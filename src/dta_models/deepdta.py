import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from src.metrics import mse, ci, rmse, r2

matplotlib.use('Agg')


class DeepDTA:
    def __init__(self, max_smi_len, max_prot_len, chem_vocab_size, prot_vocab_size,
                 embedding_dim=128, optimizer='adam', learning_rate=0.001, batch_size=256, n_epochs=200,
                 num_filters=32, smi_filter_len=4, prot_filter_len=6,
                 lm_ligand_embed_size=None, lm_protein_embed_size=None, scaler=None, **kwargs):

        print('DeepDTA: Building DeepDTA')
        self.max_smi_len = max_smi_len
        self.max_prot_len = max_prot_len
        self.chem_vocab_size = chem_vocab_size
        self.prot_vocab_size = prot_vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.smi_filter_len = smi_filter_len
        self.prot_filter_len = prot_filter_len
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.scaler = scaler
        self.lm_ligand_embed_size = lm_ligand_embed_size
        self.lm_protein_embed_size = lm_protein_embed_size

        self.model = self.__build()
        print('DeepDTA: DeepDTA is compiled')

    @classmethod
    def from_file(cls, path):
        with open(f'{path}/params.json') as f:
            dct = json.load(f)
        instance = cls(**dct)

        instance.model = tf.keras.models.load_model(f'{path}/model')

        with open(f'{path}/history.json') as f:
            instance.history = json.load(f)
        return instance

    def __build(self):
        if self.lm_ligand_embed_size is None:
            # Inputs
            chemicals = Input(shape=(self.max_smi_len,), dtype='int32')

            # Chemical encoding
            encode_smiles = Embedding(input_dim=self.chem_vocab_size + 1, output_dim=self.embedding_dim,
                                      input_length=self.max_smi_len, mask_zero=True)(chemicals)
            encode_smiles = Conv1D(filters=self.num_filters, kernel_size=self.smi_filter_len, activation='relu',
                                   padding='valid', strides=1)(encode_smiles)
            encode_smiles = Conv1D(filters=self.num_filters * 2, kernel_size=self.smi_filter_len, activation='relu',
                                   padding='valid', strides=1)(encode_smiles)
            encode_smiles = Conv1D(filters=self.num_filters * 3, kernel_size=self.smi_filter_len, activation='relu',
                                   padding='valid', strides=1)(encode_smiles)
            encode_smiles = GlobalMaxPooling1D()(encode_smiles)
        else:
            chemicals = Input(shape=(self.lm_ligand_embed_size,), dtype='float32')
            encode_smiles = chemicals

        if self.lm_protein_embed_size is None:
            # Protein encoding
            proteins = Input(shape=(self.max_prot_len,), dtype='int32')
            encode_protein = Embedding(input_dim=self.prot_vocab_size + 1, output_dim=self.embedding_dim,
                                       input_length=self.max_prot_len, mask_zero=True)(proteins)
            encode_protein = Conv1D(filters=self.num_filters, kernel_size=self.prot_filter_len, activation='relu',
                                    padding='valid', strides=1)(encode_protein)
            encode_protein = Conv1D(filters=self.num_filters * 2, kernel_size=self.prot_filter_len, activation='relu',
                                    padding='valid', strides=1)(encode_protein)
            encode_protein = Conv1D(filters=self.num_filters * 3, kernel_size=self.prot_filter_len, activation='relu',
                                    padding='valid', strides=1)(encode_protein)
            encode_protein = GlobalMaxPooling1D()(encode_protein)
        else:
            proteins = Input(shape=(self.lm_protein_embed_size,), dtype='float32')
            encode_protein = proteins

        encode_interaction = tf.keras.layers.concatenate([encode_smiles, encode_protein], axis=-1)

        # Fully connected
        FC1 = Dense(1024, activation='relu')(encode_interaction)
        FC1 = Dropout(0.1)(FC1)
        if self.lm_protein_embed_size is None:
            FC1 = Dense(1024, activation='relu')(FC1)
            FC1 = Dropout(0.1)(FC1)
        FC3 = Dense(512, activation='relu')(FC1)
        predictions = Dense(1, kernel_initializer='normal')(FC3)

        if self.optimizer == 'adam':
            opt = Adam(self.learning_rate)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(self.learning_rate)
        else:
            raise Exception('Wrong optimizer!')

        interactionModel = Model(inputs=[chemicals, proteins], outputs=[predictions])
        interactionModel.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])

        return interactionModel

    def __cast_data(data):
        return np.array(data[0]), np.array(data[1]), data[2].values

    def train(self, train_data, val_data=None, sample_weights=None, decay_type=None):
        chem_train, prot_train, y_train = DeepDTA.__cast_data(train_data)

        validation_data = None
        early_stopper = None
        if val_data is not None:
            chem_val, prot_val, y_val = DeepDTA.__cast_data(val_data)
            validation_data = ([chem_val, prot_val], y_val)
            early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)

        if decay_type is not None:
            assert early_stopper is None
            self.history = None

            for e in range(self.n_epochs):
                alpha = 0.0
                sample_weights_current = None
                if decay_type == 'lin_decrease':
                    sample_weights_current = 1 + e * (sample_weights - 1) / self.n_epochs
                elif decay_type == 'lin_increase':
                    sample_weights_current = sample_weights + e * (1 - sample_weights) / self.n_epochs

                self.history_current = self.model.fit(x=[chem_train, prot_train],
                                                      y=y_train,
                                                      sample_weight=sample_weights_current,
                                                      validation_data=validation_data,
                                                      batch_size=self.batch_size,
                                                      epochs=1).history
                print('epoch iteration: ', e)
                if self.history is None:
                    self.history = self.history_current
                else:
                    for k, _ in self.history.items():
                        self.history[k].append(self.history_current[k][0])
            return self.history

        else:
            self.history = self.model.fit(x=[chem_train, prot_train],
                                          y=y_train,
                                          validation_data=validation_data,
                                          batch_size=self.batch_size,
                                          epochs=self.n_epochs,
                                          callbacks=early_stopper).history
        print('DeepDTA: Training is complete.')
        return self.history

    def evaluate(self, evaluation_data, savedir=None, mode='train'):
        print('DeepDTA: Starting evaluation')
        results = {}
        predictions = {}
        for k, v in evaluation_data.items():
            chem_eval, prot_eval, y_eval = DeepDTA.__cast_data(v)
            preds = self.model.predict([chem_eval, prot_eval])[:, 0]
            if self.scaler:
                preds = self.scaler.inverse_transform(preds)
                y_eval = self.scaler.inverse_transform(y_eval)
            results[k] = {}
            predictions[k] = [float(pred) for pred in preds.tolist()]

            results[k]['mse'] = mse(y_eval, preds)
            if mode == 'test':
                results[k]['ci'] = ci(y_eval, preds)
                results[k]['rmse'] = rmse(y_eval, preds)
                results[k]['r2'] = r2(y_eval, preds)

        if savedir is not None:
            with open(f'{savedir}/scores.json', 'w') as f:
                json.dump(results, f, indent=4)
            with open(f'{savedir}/preds.json', 'w') as f:
                json.dump(predictions, f, indent=4)

        print('DeepDTA: Evaluation is done')
        return results

    def save(self, path):
        print('DeepDTA: Saving the model')
        self.model.save(f'{path}model')

        with open(f'{path}history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

        donot_copy = {'model', 'history', 'scaler'}
        dct = {k: v for k, v in self.__dict__.items() if k not in donot_copy}
        with open(f'{path}params.json', 'w') as f:
            json.dump(dct, f, indent=4)

    def plot_model(self, path):
        tf.keras.utils.plot_model(self.model,
                                  to_file=f'{path}model.png',
                                  show_shapes=True,
                                  show_layer_names=True,
                                  dpi=256)

    def plot_loss(self, savedir=None):
        plt.figure()
        plt.plot(self.history['loss'], label='Training Loss')
        title = 'Training Loss vs Epoch'
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
            title = 'Training and Validation Loss vs Epoch'

        plt.title(title)
        plt.ylabel('Loss - MSE')
        plt.xlabel('Epoch')
        plt.legend()

        if savedir is not None:
            plt.savefig(f'{savedir}/training_loss.png')
            plt.close()
        else:
            plt.show()
