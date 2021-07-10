from src.dta_models import DeepDTA


class BPEDeepDTA:
    def __init__(self, model_configs, bpe_tokenizer, smi_encoding_vocab_path):
        self.model = DeepDTA(**model_configs)
        self.bpe_tokenizer = bpe_tokenizer
        self.smi_encoding_vocab_path = smi_encoding_vocab_path

    def train(self, train_data, val_data=None, sample_weights=None, decay_type=None):
        pp_train = self.bpe_tokenizer.fn_pp(data=train_data.copy(),
                                            max_smi_len=self.model.max_smi_len,
                                            max_prot_len=self.model.max_prot_len,
                                            smi_encoding_vocab_path=self.smi_encoding_vocab_path)
        pp_val = None
        if val_data is not None:
            pp_val = self.bpe_tokenizer.fn_pp(data=val_data.copy(),
                                              max_smi_len=self.model.max_smi_len,
                                              max_prot_len=self.model.max_prot_len,
                                              smi_encoding_vocab_path=self.smi_encoding_vocab_path)
        return self.model.train(pp_train, pp_val, sample_weights, decay_type)

    def evaluate(self, evaluation_data, savedir=None, mode='train'):
        pp_test = {name: self.bpe_tokenizer.fn_pp(data=fold,
                                                  max_smi_len=self.model.max_smi_len,
                                                  max_prot_len=self.model.max_prot_len,
                                                  smi_encoding_vocab_path=self.smi_encoding_vocab_path)
                   for name, fold in evaluation_data.items()}
        return self.model.evaluate(pp_test, savedir, mode)

    def save(self, savedir):
        self.model.save(savedir)

    def plot_loss(self, savedir):
        self.model.plot_loss(savedir)
