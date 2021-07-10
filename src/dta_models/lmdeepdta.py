from src.dta_models import DeepDTA


class LMDeepDTA:
    def __init__(self, model_configs, lm_tokenizer, smi_encoding_vocab_path):
        self.model = DeepDTA(**model_configs)
        self.lm_tokenizer = lm_tokenizer

    def train(self, train_data, val_data=None, sample_weights=None, decay_type=None):
        pp_train = self.lm_tokenizer.fn_pp(data=train_data.copy())
        pp_val = None
        if val_data is not None:
            pp_val = self.lm_tokenizer.fn_pp(data=val_data.copy())
        return self.model.train(pp_train, pp_val, sample_weights, decay_type)

    def evaluate(self, evaluation_data, savedir=None, mode='train'):
        pp_test = {name: self.lm_tokenizer.fn_pp(data=fold)
                   for name, fold in evaluation_data.items()}
        return self.model.evaluate(pp_test, savedir, mode)

    def save(self, savedir):
        self.model.save(savedir)

    def plot_loss(self, savedir):
        self.model.plot_loss(savedir)
