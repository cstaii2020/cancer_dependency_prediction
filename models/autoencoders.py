import pickle

import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from models.neural_net_models import NeuralNetAbstractModel


class AutoEncoderDecoderNet(nn.Module):
    def __init__(self, d_in, hs, d_encoded):
        super(AutoEncoderDecoderNet, self).__init__()
        flatten = lambda l: [item for sublist in l for item in sublist]

        enc_h = [d_in] + hs
        encoder_layers = flatten([[nn.Linear(enc_h[i], enc_h[i + 1]), nn.Tanh()] for i in range(len(enc_h) - 1)]) \
                 + [nn.Linear(enc_h[-1], d_encoded)]

        hs.reverse()
        dec_h = [d_encoded] + hs
        decoder_layers = flatten([[nn.Linear(dec_h[i], dec_h[i + 1]), nn.Tanh()] for i in range(len(dec_h) - 1)]) \
                         + [nn.Linear(dec_h[-1], d_in)]

        self.encoder = nn.Sequential(*encoder_layers).double()
        self.decoder = nn.Sequential(*decoder_layers).double()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)


class AutoEncoderModel(NeuralNetAbstractModel):
    """
    source == target
    """

    def __init__(self, source, hidden_layers, epochs, logger, target_names, fname, batch_size=32, lr=1e-4,
                 wd=0.0, group=None, is_linear=False, save_model=True):
        super().__init__(source, source, hidden_layers, epochs, logger, target_names, fname, batch_size=batch_size, lr=lr,
                 wd=wd, group=group, is_linear=is_linear, save_model=save_model)
        self.full_output = []

    def build_model(self, dim_source=None, dim_target=None):
        if dim_source is None:
            dim_source = self.source.shape[1]
        if dim_target is None:
            dim_target = self.hidden_layers[-1]

        hidden_layer = self.hidden_layers[:-1]
        return AutoEncoderDecoderNet(dim_source, hidden_layer, dim_target)

    def get_loss(self):
        return nn.MSELoss()

    def summarize_results(self):
        with open(f"{self.out_name}/full_{self.test_count}.pickle", "wb") as f:
            pickle.dump(self.full_output, f)

        self.output.to_csv(f"{self.out_name}/results_{self.test_count}.csv")
        return self.output

    def evaluate_so_far(self, model, train_source, train_target, test_source, test_target, epoch):
        reconstructed_train = model(train_source)
        out_dict = {
            "epoch": epoch,
            "reconstructed_train_mse": self.get_mse(reconstructed_train, train_target),
            "reconstructed_train_pearson": self.get_pearson_averages(reconstructed_train, train_target)
        }
        if len(test_source) != 0:
            reconstructed_test = model(test_source)
            out_dict["reconstructed_test_mse"] = self.get_mse(reconstructed_test, test_target)
            out_dict["reconstructed_test_pearson"] = self.get_pearson_averages(reconstructed_test, test_target)
            self.full_output.append({
                "epoch": epoch,
                "pearson": self.get_pearsons(reconstructed_test, test_target),
                "mse": self.get_mse(reconstructed_test, test_target, axis=1)
            })

        self.output = self.output.append(out_dict, ignore_index=True)

    def setup_output(self):
        return pd.DataFrame(columns=["epoch", "reconstructed_train_mse", "reconstructed_test_mse",
                                     "reconstructed_train_pearson", "reconstructed_test_pearson"])

    def get_encoder_decoder(self):
        return self.current_net

    @staticmethod
    def get_mse(x, y, axis=None):
        if axis is None:
            return torch.mean((x - y) ** 2).item()
        return list(torch.mean((x - y) ** 2, axis=axis).detach().numpy())

    @staticmethod
    def get_pearson_averages(vectors1, vectors2):
        return sum(AutoEncoderModel.get_pearsons(vectors1, vectors2)) / vectors1.shape[1]

    @staticmethod
    def get_pearsons(vectors1, vectors2):
        return [pearsonr(x1, x2)[0] for x1, x2 in
                    zip(vectors1.detach(), vectors2.detach())]
