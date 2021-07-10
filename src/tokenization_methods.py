import json
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel

from src.utils import encode_smiles
from src.word_identification import WordIdentifier


class SMIAwareBPE:
    def __init__(self, paths, chem_vocab_size, prot_vocab_size, **kwargs):
        if chem_vocab_size == 94 and prot_vocab_size == 26:
            self.name = 'deepdta'
        else:
            self.name = f'bpedta/chem_{chem_vocab_size}_prot_{prot_vocab_size}'
        self.chem_vocab_name = 'chembl27_enc_bpe_'
        self.prot_vocab_name = 'uniprot_bpe_'
        self.chem_vocab_size = chem_vocab_size
        self.prot_vocab_size = prot_vocab_size
        chem_vocab_path = paths['chem_vocab']
        prot_vocab_path = paths['prot_vocab']
        chem_tokenizer_path = f'{chem_vocab_path}{self.chem_vocab_name}{chem_vocab_size}.json'
        prot_tokenizer_path = f'{prot_vocab_path}{self.prot_vocab_name}{prot_vocab_size}.json'
        self.chem_tokenizer = WordIdentifier.from_file(chem_tokenizer_path)
        self.prot_tokenizer = WordIdentifier.from_file(prot_tokenizer_path)

    def fn_pp(self, **kwargs):
        data = kwargs['data']
        max_smi_len = kwargs['max_smi_len']
        max_prot_len = kwargs['max_prot_len']
        encoding_vocab_path = kwargs['smi_encoding_vocab_path']
        data['smiles'] = data['smiles'].apply(encode_smiles, encoding_vocab_path=encoding_vocab_path)
        chemicals = self.chem_tokenizer.identify_words(data['smiles'], padding_len=max_smi_len, out_type='int')
        proteins = self.prot_tokenizer.identify_words(data['aa_sequence'], padding_len=max_prot_len, out_type='int')
        return chemicals, proteins, data['affinity_score']


class ProteinTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    def tokenize(self, sequence, only_tokens=False):
        spaced_seq = " ".join(sequence)
        sequence = re.sub(r"[UZOB]", "X", spaced_seq)
        encoded_input = self.tokenizer(sequence, return_tensors='pt')
        if not (only_tokens):
            return encoded_input
        return list(encoded_input["input_ids"].squeeze(0).detach().cpu().numpy())


class LigandTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    def tokenize(self, smiles, only_tokens=False):
        tokens = self.tokenizer(smiles)
        if not (only_tokens):
            return tokens
        return list(np.array(tokens['input_ids'], dtype="int"))


class LMTokenizer:
    def __init__(self, db, tokenizer):
        self.database = db
        self.tokenizer = tokenizer

    def identify_words(self, sequences, padding_len=None, out_type='int'):
        words = []
        for seq in sequences:
            if seq in self.database['seq2tokens']:
                words.append(self.database['seq2tokens'][seq])
            else:
                try:
                    tokens = self.tokenizer.tokenize(seq)
                    words.append(tokens)
                    self.database['seq2tokens'][seq] = tokens
                except:
                    continue

        with open(self.database_path, 'w') as f:
            json.dump(self.database, f)
        return words

    def tokenize(self, sequence, only_tokens):
        return self.tokenizer.tokenize(sequence, only_tokens)


class LanguageModel:
    def __init__(self, paths, chem_vocab_size, prot_vocab_size, **kwargs):
        self.name = 'lmdta'
        self.chem_vocab_size = "NA"
        self.prot_vocab_size = "NA"

        self.lig_embedder = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.lig_embedder.eval()
        self.prot_embedder = BertModel.from_pretrained("Rostlab/prot_bert")
        self.prot_embedder.eval()

        self.lm_protein_path = kwargs["lm_protein_path"]
        self.lm_ligand_path = kwargs["lm_ligand_path"]

        if os.path.exists(self.lm_protein_path):
            with open(self.lm_protein_path, "r") as f:
                self.prot_dict = json.load(f)
        else:
            self.prot_dict = {'seq2tokens': {}, 'seq2features': {}}
        if os.path.exists(self.lm_ligand_path):
            with open(self.lm_ligand_path, "r") as f:
                self.lig_dict = json.load(f)
        else:
            self.lig_dict = {'seq2tokens': {}, 'seq2features': {}}
        self.chem_tokenizer = LMTokenizer(self.lig_dict, LigandTokenizer())
        self.prot_tokenizer = LMTokenizer(self.prot_dict, ProteinTokenizer())

    def update_database(self, path, data):
        for k, v in data['seq2features'].items():
            if isinstance(v, np.ndarray):
                data['seq2features'][k] = v.tolist()
        for k, v in data['seq2tokens'].items():
            if isinstance(v, np.ndarray):
                data['seq2tokens'][k] = v.tolist()
        with open(path, 'w') as f:
            json.dump(data, f)

    def embed_protein(self, sequence):
        encoded_input = self.prot_tokenizer.tokenize(sequence, False)
        output = self.prot_embedder(**encoded_input)
        return output["last_hidden_state"].squeeze(0).detach().cpu().numpy().mean(0)

    def embed_ligand(self, smiles):
        tokens = self.chem_tokenizer.tokenize(smiles, False)
        input_ligand = torch.LongTensor([tokens['input_ids']])
        output = self.lig_embedder(input_ligand, return_dict=True)
        return torch.mean(output.last_hidden_state[0], axis=0).cpu().detach().numpy()

    def fn_pp(self, **kwargs):
        data = kwargs["data"]
        embed_smiles = []
        embed_prots = []
        affinity_scores = []
        for smiles, prot, score in zip(data["smiles"], data["aa_sequence"], data["affinity_score"]):
            if (smiles in self.lig_dict['seq2features']) and (prot not in self.prot_dict['seq2features']):
                try:
                    self.prot_dict['seq2features'][prot] = self.embed_protein(prot)
                except:
                    continue
            elif (smiles not in self.lig_dict['seq2features']) and (prot in self.prot_dict['seq2features']):
                try:
                    self.lig_dict['seq2features'][smiles] = self.embed_ligand(smiles)
                except:
                    continue
            elif (smiles not in self.lig_dict['seq2features']) and (prot not in self.prot_dict['seq2features']):
                self.prot_dict['seq2features'][prot] = self.embed_protein(prot)
                self.lig_dict['seq2features'][smiles] = self.embed_ligand(smiles)
            embed_smiles.append(self.lig_dict['seq2features'][smiles])
            embed_prots.append(self.prot_dict['seq2features'][prot])
            affinity_scores.append(score)

        self.update_database(self.lm_ligand_path, self.lig_dict)
        self.update_database(self.lm_protein_path, self.prot_dict)
        embed_prots = np.array(embed_prots, dtype=np.float32)
        embed_smiles = np.array(embed_smiles, dtype=np.float32).reshape(len(embed_prots), -1)
        affinity_scores = np.array(affinity_scores, dtype=np.float32)

        return embed_smiles, embed_prots, pd.Series(affinity_scores)
