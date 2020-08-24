import functools
import os
import time
from abc import ABC, abstractmethod
import pandas as pd
import pickle
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from cca import column_correlation
from models.base_models import LearningModel, get_binary_evaluation, get_binary_evaluation_many_targets
from models.cca import tutorial_on_cca
from models.model_utils import normalize_by_train


"""
              Network Objects
"""


class DeepNet(nn.Module):
    def __init__(self, d_in, hs, d_out, sigmoid=False, linear=False):
        super(DeepNet, self).__init__()
        flatten = lambda l: [item for sublist in l for item in sublist]

        activation = nn.Tanh

        hs = [d_in] + hs
        layers = flatten([[nn.Linear(hs[i], hs[i + 1]), activation()] for i in range(len(hs) - 1)])

        if sigmoid:
            layers[-1] = nn.Sigmoid()

        if linear:
            self.seq_nn = nn.Sequential(
                nn.Linear(d_in, d_out)
            ).double()
        else:
            self.seq_nn = nn.Sequential(
                *(layers + [nn.Linear(hs[-1], d_out)])
            ).double()

    def forward(self, x):
        return self.seq_nn(x)


class OneToOneDeepNet(nn.Module):
    def __init__(self, d_in, hs, d_out, sigmoid=False, linear=False):
        super(OneToOneDeepNet, self).__init__()
        flatten = lambda l: [item for sublist in l for item in sublist]
        sequences_nn_list = []

        for i in range(d_out):
            layers = flatten([[nn.Linear(1, 1), nn.Tanh()] for i in range(len(hs) - 1)])

            if sigmoid:
                layers[-1] = nn.Sigmoid()

            if linear:
                seq_nn = nn.Sequential(
                    nn.Linear(1, 1)
                ).double()
            else:
                seq_nn = nn.Sequential(
                    *(layers + [nn.Linear(1, 1)])
                ).double()

            sequences_nn_list.append(seq_nn)

        self.seq_nn = nn.ModuleList(sequences_nn_list)

    def forward(self, x):
        new_x = torch.zeros_like(x)
        for i in range(x.shape[1]):
            new_x[:, i] = self.seq_nn[i](x[:, i].reshape(-1, 1)).reshape(-1)

        return new_x


class DeepGroupNet(nn.Module):
    def __init__(self, mid, dout, s_groups, t_groups):
        super(DeepGroupNet, self).__init__()

        assert not hasattr(mid, '__iter__'), "has to be only one middle layer in group neural net"

        self.s_groups = s_groups
        self.t_groups = t_groups
        self.dout = dout

        self.s_group_layers = {}
        total_out = 0
        for group_key, group in s_groups.items():
            out = 2  # max(1, int(len(group) / 2))
            self.s_group_layers[group_key] = (nn.Linear(len(group), out), nn.Tanh())
            total_out += out

        self.middle_layers = nn.Sequential(nn.Linear(total_out, mid), nn.Tanh(), nn.Linear(mid, dout), nn.Tanh())

        # self.t_group_layers = {}
        #
        # for group_key, group in t_groups.items():
        #     self.t_group_layers[group_key] = (nn.Linear(mid, len(group)), nn.Tanh())

        self.final = nn.Linear(dout, dout)

    def forward(self, x):
        x = x.float()
        processed = []
        for gk, group in self.s_groups.items():
            linear, activation = self.s_group_layers[gk]
            processed.append(activation(linear(x[:, group])))

        x = torch.cat(processed, dim=1)
        x = self.middle_layers(x)

        # new_x = torch.zeros((x.shape[0], self.dout))
        # for gk, group in self.t_groups.items():
        #     linear, activation = self.t_group_layers[gk]
        #     new_x[:, group] = activation(linear(x))

        x = self.final(x)
        return x.double()


class NeuralNetAbstractModel(LearningModel, ABC):
    def __init__(self, source, target, hidden_layers, epochs, logger, target_names, fname, batch_size=32, lr=1e-4,
                 wd=0.0, group=None, is_linear=False, source_autoencoder=None, target_autoencoder=None, adamw=False,
                 model_type=DeepNet, comp=None, use_scikit=None, halt=False, save_model=False):
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)

        if source_autoencoder is not None:
            source = source_autoencoder.encode(torch.as_tensor(source)).detach().numpy()

        self.original_target_tensor = torch.as_tensor(target)
        if target_autoencoder is not None:
            target = target_autoencoder.encode(torch.as_tensor(target)).detach().numpy()

        super().__init__(source, target, fname, target_encoder=None, class_names=None)

        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.logger = logger
        self.group = group
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.target_names = target_names
        self.output = self.setup_output()
        self.test_count = 0
        self.is_linear = is_linear
        self.current_net = None
        self.target_autoencoder = target_autoencoder
        self.model_type = model_type
        self.adamw = adamw
        self.comp = comp
        self.use_scikit = use_scikit
        self.halt = halt
        self.predictions = None
        self.save_model = save_model


        original_fname = f"{fname}_e{self.epochs}_b{self.batch_size}_l{self.lr}_d{self.hidden_layers}_wd_{self.wd}"
        if comp is not None:
            original_fname = f"{original_fname}_comp_{comp}"
        if use_scikit is not None:
            original_fname = f"{original_fname}_{use_scikit}"

        new_fname = original_fname
        count = 1
        while os.path.isfile(new_fname + ".csv") or os.path.isdir(new_fname):
            new_fname = original_fname + f"_({count})"
            count += 1

        self.out_name = new_fname
        os.mkdir(self.out_name)

    @abstractmethod
    def setup_output(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    def build_model(self, dim_source=None, dim_target=None):
        if dim_source is None:
            dim_source = self.source.shape[1]
        if dim_target is None:
            dim_target = self.target.shape[1]

        if self.group is None:
            model = self.model_type(dim_source, self.hidden_layers, dim_target, linear=self.is_linear)
        else:
            target_group, source_group = self.group
            model = DeepGroupNet(self.hidden_layers, dim_target, source_group, target_group)

        return model

    @staticmethod
    def train_episode(model, loss_func, optimizer, x, y):
        y_predicted = model(x)
        loss = loss_func(y_predicted, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return y_predicted, loss

    def train_and_save(self):
        self.train_eval(np.array(range(self.source.shape[0])), np.array([]))

    def train_eval(self, train_index, test_index):
        self.test_count += 1
        start = time.time()

        _, normalized_source = normalize_by_train(self.source[train_index], self.source)

        if self.comp is not None:
            if self.use_scikit is not None:
                if self.use_scikit == 'cca':
                    dim_reduction = CCA(n_components=self.comp)
                else:
                    dim_reduction = PCA(n_components=self.comp)
                # fit cca according to train data only
                dim_reduction.fit(normalized_source[train_index], self.target[train_index])
                # convert source into lower dimensional representation
                normalized_source = dim_reduction.transform(normalized_source)
            else:
                _, wa, _ = tutorial_on_cca(normalized_source[train_index], self.target[train_index])
                normalized_source = normalized_source @ wa[:, :self.comp]

        model = self.build_model(dim_source=normalized_source.shape[1])
        self.current_net = model

        dataset = TensorDataset(torch.from_numpy(normalized_source), torch.from_numpy(self.target).double())
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_index))

        loss_func = self.get_loss()
        if self.adamw:
            self.logger.info("ADAM W !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
            adam_optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            adam_optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        max_epochs = max(self.epochs)

        train_source, train_target = dataset[train_index]
        test_source, test_target = dataset[test_index]

        if self.target_autoencoder is not None:
            train_target = self.original_target_tensor[train_index]
            test_target = self.original_target_tensor[test_index]

        max_evaluation = None
        non_improvement = 0

        for epoch in range(max_epochs):
            for batch_index, (source_sample, target_sample) in enumerate(train_loader):
                _, loss = self.train_episode(model, loss_func, adam_optimizer, source_sample, target_sample)
                print(f"epoch : {epoch + 1} / {max_epochs} , batch index : {batch_index + 1} / {len(train_loader)}"
                      f" | loss = {loss.item()}", end="\r")

            if epoch + 1 in self.epochs:
                predictor = model if self.target_autoencoder is None else\
                    lambda x: self.target_autoencoder.decode(model(x))
                evaluation = self.evaluate_so_far(predictor, train_source, train_target, test_source, test_target, epoch)

                if evaluation is not None and self.halt:
                    if max_evaluation is None:
                        max_evaluation = evaluation
                    elif max_evaluation < evaluation:
                        max_evaluation = evaluation
                        non_improvement = 0
                    else:
                        non_improvement += 1

                    if non_improvement == 5:
                        break

        print('')

        self.predictions = predictor(train_source)

        result_summary = self.summarize_results()
        if len(test_index) == 0 or self.save_model:
            i = 1
            while os.path.isfile(f"{self.out_name}/model_{i}.pickle"):
                i += 1
            with open(f"{self.out_name}/model_{i}.pickle", "wb") as f:
                pickle.dump(model, f)

        end = time.time()
        self.logger.info(f"model took {end - start} seconds")
        self.output = self.setup_output()
        return result_summary

    @abstractmethod
    def summarize_results(self):
        pass

    def combined_eval(self, evaluations):
        average_results = pd.concat(evaluations).groupby(level=0).mean()
        average_results.to_csv(f"{self.out_name}/average_results.csv")

        return average_results

    @abstractmethod
    def evaluate_so_far(self, model, train_source, train_target, test_source, test_target, epoch):
        pass


class NeuralNetClassifier(NeuralNetAbstractModel):
    def __init__(self, target, fname, threshold=0.65, **kwargs):
        super().__init__(target=np.vectorize(lambda x: 0 if x < threshold else 1)(target),
                         fname=f"{fname}_BCE_convert", **kwargs)
        self.threshold = threshold

    def get_loss(self):
        return torch.nn.BCEWithLogitsLoss()
        # return torch.nn.MSELoss()

    # def build_model(self):
    #     self.logger.info(f"model with sigmoid")
    #     dim_source = self.source.shape[1]
    #     dim_target = self.target.shape[1]
    #     return DeepNet(dim_source, self.hidden_layers, dim_target, sigmoid=True)

    def evaluate_so_far(self, model, train_source, train_target, test_source, test_target, epoch):
        # train_c_matrix, train_auc, train_precision, train_recall, train =
        # self.get_classification_eval(model, train_source, train_target)
        # test_matrix, test_auc, test_precision, test_recall, test_accuracy = \
        #     self.get_classification_eval(model, test_source, test_target)
        # new_result = pd.DataFrame({
        #     'target': self.target_names, 'epoch': [epoch for x in self.target_names],
        #     'TP': test_matrix[:, 3], 'TN': test_matrix[:, 0], 'FP': test_matrix[:, 1], 'FN': test_matrix[:, 2],
        #     'AUC-ROC': test_auc, 'precision': test_precision, 'recall': test_recall, 'accuracy': test_accuracy})
        # self.output = self.output.append(new_result)

        # One target
        # eval_dict = self.get_classification_eval(model, test_source, test_target)
        # new_result = pd.Series(eval_dict, name=epoch+1)
        # self.output = self.output.append(new_result)

        # Many targets
        if not os.path.isdir(f"{self.out_name}/results_{self.test_count}"):
            os.mkdir(f"{self.out_name}/results_{self.test_count}")

        results_df = self.get_classification_eval(model, test_source, test_target)
        results_df.to_csv(f"{self.out_name}/results_{self.test_count}/epoch_{epoch}.csv")

        average_result = results_df.loc['average']
        average_result.name = epoch
        self.output = self.output.append(average_result)

    def setup_output(self):
        # one target
        return pd.DataFrame(columns=["epoch", "TP", "TN", "FP", "FN", "AUC-ROC", "precision", "recall"])\
            .set_index("epoch")
        # TODO : multi-target
        # return []

    def summarize_results(self):
        # one target:
        # self.output.to_csv(f"{self.out_name}/results_{self.test_count}.csv")
        # return self.output
        # TODO : multi-target
        # for epoch, result_df in self.output.items()
        #     result_df['AUC-ROC']
        return self.output

    def apply_threshold(self, prediction_probabilities):
        return np.vectorize(lambda x: 0 if x < self.threshold else 1)(prediction_probabilities)

    def get_classification_eval(self, model, source, target):
        predicted = model(source)
        prediction_probabilities = predicted.detach().numpy()
        actual = self.apply_threshold(target.detach().numpy())
        predictions = self.apply_threshold(prediction_probabilities)
        return get_binary_evaluation_many_targets(actual, predictions, prediction_probabilities, self.target_names)
        # return np.array([list(confusion_matrix(actual, predicted)) for predicted, actual in
        #         zip(all_predictions.transpose(), all_actual.transpose())]), \
        #        [roc_auc_score(actual, predicted) for predicted, actual in
        #         zip(all_predictions.transpose(), all_actual.transpose())], \
        #        [precision_score(actual, predicted) for predicted, actual in
        #         zip(all_predictions.transpose(), all_actual.transpose())], \
        #        [recall_score(actual, predicted) for predicted, actual in
        #         zip(all_predictions.transpose(), all_actual.transpose())], \
        #        [accuracy_score(actual, predicted) for predicted, actual in
        #         zip(all_predictions.transpose(), all_actual.transpose())]

    # def combined_eval(self, evaluations):
    #     epochs_results = {}
    #     for test in evaluations:
    #         for i, e in enumerate(self.epochs):
    #             if e not in epochs_results:
    #                 epochs_results[e] = []
    #
    #             epochs_results[e].append(test[i])
    #
    #     os.mkdir(f"{self.out_name}/results_average")
    #     for epoch, epoch_results in epochs_results.items():
    #         average_epoch_results = pd.concat(epoch_results).groupby(level=0).mean()
    #         average_epoch_results.to_csv(f"{self.out_name}/results_average/epoch_{epoch}")



class NeuralNetRegressor(NeuralNetAbstractModel):
    @staticmethod
    def get_ranges():
        return [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1)]

    def get_loss(self):
        return torch.nn.MSELoss()

    def setup_output(self):
        ranges_col_names = [f"{r1}-{r2}" for r1, r2 in self.get_ranges()]
        return pd.DataFrame(index=list(self.target_names) + ['mean_squared_error']), \
               pd.DataFrame(index=['avg_test_sig', 'avg_train_sig', 'test_significance_rate',
                                   'avg_test', 'avg_train', 'mse_test', 'mse_train'] + ranges_col_names + ["total"]), \
               pd.DataFrame(index=ranges_col_names)

    def evaluate_so_far(self, model, train_source, train_target, test_source, test_target, epoch):
        _, _, train_pearson, train_mse = self.get_evaluation(model, train_source, train_target)
        test_actual, test_predictions, test_pearson, test_mse = self.get_evaluation(model, test_source, test_target)
        significant_test_pearson = self.get_significant(test_pearson)
        significant_train_pearson = self.get_significant(train_pearson)
        test_pearson_values = self.get_pearsons_value(test_pearson)
        train_pearson_values = self.get_pearsons_value(train_pearson)

        pearson_details, general_results, ranges_details = self.output

        pearson_details.insert(pearson_details.shape[1], f'train {epoch + 1}', train_pearson_values + [train_mse])
        pearson_details.insert(pearson_details.shape[1], f'test {epoch + 1}', test_pearson_values + [test_mse])

        predictable_targets = [pearson_details[pearson_details[f'test {epoch + 1}'].between(r1, r2)].index for (r1, r2)
                               in self.get_ranges()]
        predictable_targets_count = [ts.shape[0] for ts in predictable_targets]

        general_results.insert(general_results.shape[1], epoch + 1,
                               [avg(significant_test_pearson), avg(significant_train_pearson),
                                len(significant_test_pearson) / len(test_pearson)]
                               + [avg(test_pearson_values), avg(train_pearson_values), test_mse, train_mse]
                               + predictable_targets_count + [sum(predictable_targets_count)])

        ranges_details.insert(ranges_details.shape[1], epoch + 1,
                              [", ".join(targets) for targets in predictable_targets])
        # TODO: check if it self.output needs to be reassigned to those values
        return avg(test_pearson_values)

    def summarize_results(self):
        dir_name = self.out_name
        pearson_details, general_results, ranges_details = self.output

        # with open(f"{dir_name}/pearson_details_{self.test_count}.pickle", 'wb') as f:
        #     pickle.dump(pearson_details, f)
        pearson_details.to_csv(f"{dir_name}/pearson_details_{self.test_count}.csv")

        # with open(f"{dir_name}/ranges_details_{self.test_count}.pickle", 'wb') as f:
        #     pickle.dump(ranges_details, f)
        ranges_details.to_csv(f"{dir_name}/ranges_details_{self.test_count}.csv")

        general_results.to_csv(f"{dir_name}/results_{self.test_count}.csv")

        # with open(f"{dir_name}/model_{self.test_count}.pickle", 'wb') as f:
        #     pickle.dump(self.model, f)

        return pearson_details

    def combined_eval(self, evaluations):
        out_name = self.out_name
        average_pearson_details = pd.concat(evaluations).groupby(level=0).mean()
        mse = average_pearson_details.loc["mean_squared_error"]
        average_pearson_details = average_pearson_details.iloc[:-1]
        average_pearson_details.to_csv(f"{out_name}/average_pearson_details.csv")
        ranges_col_names = [f"{r1}-{r2}" for r1, r2 in self.get_ranges()]

        general_results = pd.DataFrame(columns=average_pearson_details.columns,
                                       index=ranges_col_names+["total", "pearson_average", "mse"])
        ranges_details = pd.DataFrame(columns=average_pearson_details.columns,
                                       index=ranges_col_names)

        for c in average_pearson_details.columns:
            predictable_targets = [average_pearson_details[average_pearson_details[c].between(r1, r2)].index
                                   for (r1, r2) in self.get_ranges()]
            ranges_details.loc[:, c] = [", ".join(t) for t in predictable_targets]

            predictable_targets_count = [ts.shape[0] for ts in predictable_targets]
            total_predictable_targets_count = sum(predictable_targets_count)
            average_pearson = average_pearson_details.loc[:, c].mean()

            general_results.loc[:, c] = predictable_targets_count +\
                                        [total_predictable_targets_count, average_pearson, mse[c]]

        general_results.to_csv(f"{out_name}/average_results.csv")
        ranges_details.to_csv(f"{out_name}/ranges_details.csv")

        # return the best average pearson test score (::-2)
        best_test = np.argmax(general_results.loc["pearson_average", ::-2])
        best_value = general_results.loc["pearson_average", ::-2][best_test]
        self.logger.info(f"max epoch : {best_test}.   Got {best_value}")
        return best_value

    @staticmethod
    def get_evaluation(model, source, target):
        predicted = model(source)
        all_predictions = predicted.detach().numpy()
        # SHUFFLING
        # np.random.shuffle(all_predictions)
        all_actual = target.detach().numpy()
        mse = mean_squared_error(all_predictions, all_actual)
        correlations = [pearsonr(actual, predicted) for predicted, actual
                                             in zip(all_predictions.transpose(), all_actual.transpose())]
        return all_actual, all_predictions, correlations, mse

    @staticmethod
    def get_significant(pearson):
        return [p[0] for p in pearson if p[1] < 0.05]

    @staticmethod
    def get_pearsons_value(pearson):
        return [p[0] for p in pearson]


"""
        Helper func
"""


def avg(a):
    if len(a) == 0:
        return None
    return sum(a) / len(a)
