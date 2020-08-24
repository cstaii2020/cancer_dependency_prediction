import os
import numpy as np
import torch
from sklearn.base import clone
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    mean_squared_error
import pandas as pd

from models.cca import tutorial_on_cca
from models.model_utils import normalize_by_train
from scipy.stats import pearsonr


class LearningModel(ABC):
    def __init__(self, source, target, out_name, target_encoder=None, class_names=None):
        self.source = source
        self.out_name = out_name
        self.target_encoder = target_encoder
        self.original_target = target
        self.class_names = class_names

        if target_encoder is not None:
            self.target = target_encoder.encode(torch.as_tensor(target)).detach().numpy()
        else:
            self.target = target

    def set_output(self, out_name):
        self.out_name = out_name

    @abstractmethod
    def train_eval(self, train_index, test_index):
        pass

    @abstractmethod
    def combined_eval(self, evaluations):
        pass


class Regressor(LearningModel):
    def __init__(self, source, target, reg, target_names, out_name, comp=None, use_scikit=None):
        super().__init__(source, target, out_name)
        self.reg = reg
        self.target_names = target_names
        self.targets_count = len(target_names)
        self.out_name = out_name
        self.comp = comp
        self.use_scikit = use_scikit
        # self.output = pd.DataFrame(columns=["TP", "TN", "FP", "FN", "AUC-ROC", "precision", "recall"])

    def build_model(self):
        if isinstance(self.reg, IdentityModel):
            return self.reg

        return clone(self.reg)

    def train_eval(self, train_index, test_index, ignore_eval=False):
        normalized_train, normalized_test = normalize_by_train(self.source[train_index], self.source[test_index])

        if self.comp is not None:
            if self.use_scikit is not None:
                if self.use_scikit == 'cca':
                    dim_reduction = CCA(n_components=self.comp)
                else:
                    dim_reduction = PCA(n_components=self.comp)
                # fit cca according to train data only
                dim_reduction.fit(normalized_train, self.target[train_index])
                # convert source into lower dimensional representation
                normalized_train = dim_reduction.transform(normalized_train)
                normalized_test = dim_reduction.transform(normalized_test)
            else:
                _, wa, _ = tutorial_on_cca(normalized_train, self.target[train_index])
                normalized_train = normalized_train @ wa[:, :self.comp]
                normalized_test = normalized_test @ wa[:, :self.comp]

        model = self.build_model()


        model.fit(normalized_train, self.target[train_index])

        prediction = model.predict(normalized_test)

        # res_df.to_csv(f"{self.out_name}/res1.csv")
        if not ignore_eval:
            return self.evaluate_regression(prediction, test_index)
        else:
            return prediction

    def evaluate_regression(self, prediction, test_index):
        pearson_list = [pearsonr(actual, predicted)[0] for predicted, actual in
                        zip(prediction.transpose(), self.target[test_index].transpose())]
        mse_list = [mean_squared_error(actual, predicted) for predicted, actual in
                    zip(prediction.transpose(), self.target[test_index].transpose())]

        average_pearson = sum(pearson_list) / len(pearson_list)
        average_mse = sum(mse_list) / len(mse_list)

        return pd.DataFrame(columns=list(self.target_names) + ['average'],
                              data=[pearson_list + [average_pearson], mse_list + [average_mse]],
                              index=["pearson", "mse"])

    def combined_eval(self, evaluations):
        all_evaluations = pd.concat(evaluations)
        average_results = all_evaluations.groupby(level=0).mean()

        average_mse = average_results.loc["mse"]
        average_pearson = average_results.loc["pearson"]
        average_mse.name = "mse_average"
        average_pearson.name = "pearson_average"

        combined = all_evaluations.append(average_pearson).append(average_mse)
        original_name = f"{self.out_name}_results"
        if os.path.isfile(f"{original_name}.csv"):
            i = 1
            fname = f"{original_name}_({i})"
            while os.path.isfile(f"{fname}.csv"):
                i += 1
                fname = f"{original_name}_({i})"
        else:
            fname = original_name

        combined.to_csv(f"{fname}.csv")
        # return combined.loc[["pearson_average", "mse_average"], "average"]
        return combined.loc["pearson_average", "average"]


class IdentityModel(object):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return x


class Classifier(LearningModel):
    def __init__(self, source, target, clf, class_names, out_name):
        super().__init__(source, target, out_name)
        self.clf = clf
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.output = pd.DataFrame(columns=["TP", "TN", "FP", "FN", "AUC-ROC", "precision", "recall"])

    def build_model(self):
        return clone(self.clf)

    def train_eval(self, train_index, test_index, ignore_eval=False):
        model = self.build_model()
        normalized_train, normalized_test = normalize_by_train(self.source[train_index], self.source[test_index])

        model.fit(normalized_train, self.target[train_index])

        prediction = model.predict(normalized_test)
        prediction_probabilities = model.predict_proba(normalized_test)
        actual = self.target[test_index]

        if not ignore_eval:
            if self.n_classes > 2:
                # TODO: update so that every class has all evaluations
                auc = np.zeros(self.n_classes)
                for i in range(self.n_classes):
                    new_labels = np.vectorize(lambda x: 1 if x == i else 0)(actual)
                    auc[i] = roc_auc_score(new_labels, prediction_probabilities[:, i])
            else:
                eval_dict = get_binary_evaluation(actual, prediction, prediction_probabilities[:, 1])
                new_results = pd.Series(eval_dict, name=self.output.shape[0])
                self.output = self.output.append(new_results)
                auc = eval_dict['AUC-ROC']

            return auc, actual
        else:
            return prediction

    def combined_eval(self, scores):
        avg = self.output.mean()
        avg.name = "average"
        self.output = self.output.append(avg)

        count = 0
        while os.path.isdir(f"{self.out_name}-{count}") or os.path.isfile(f"{self.out_name}-{count}.csv"):
            count += 1

        self.output.to_csv(f"{self.out_name}-{count}.csv")

        return avg


class KFoldTest:
    def __init__(self, splits, model: LearningModel, logger):
        self.splits = splits
        self.kf = KFold(n_splits=splits)
        self.model = model
        self.scores = []
        self.logger = logger

    def run_test(self):
        for test_count, (train_index, test_index) in enumerate(self.kf.split(self.model.source)):
            self.logger.info(f"running test {test_count + 1} out of {self.splits}")
            self.scores.append(self.model.train_eval(train_index, test_index))

    #TODO : make sure to return single score
    def evaluate(self):
        return self.model.combined_eval(self.scores)


class CombinedRegressor(LearningModel):
    def __init__(self, source, target, out_name, dif_classifier: Classifier, zero_classifier: Classifier,
                 reg: Regressor):
        super().__init__(source=source, target=target, out_name=out_name)
        self.dif_classifier = dif_classifier
        self.zero_classifier = zero_classifier
        self.reg = reg
        # self.dif_classifier = Classifier(source=source, target=to_differentiality_label(target),
        #                                  **dif_classifier_params)
        # self.zero_classifier = Classifier(source=source, target=to_zero_one(target),
        #                                   **zero_classifier_params)
        # if is_nn:
        #     self.reg = NeuralNetRegressor(**reg_params)
        # else:
        #     self.reg = Regressor(**reg_params)

    def train_eval(self, train_index, test_index):
        dif_classification = self.dif_classifier.train_eval(train_index, test_index,
                                                                           ignore_eval=True)
        train_labels = self.dif_classifier.target[train_index]

        train_index_dif = train_index[np.where(train_labels == 1)]
        train_index_non_dif = train_index[np.where(train_labels == 0)]
        test_index_dif = test_index[np.where(dif_classification == 1)]
        test_index_non_dif = test_index[np.where(dif_classification == 0)]

        predicted_zero_one = self.zero_classifier.train_eval(train_index_non_dif, test_index_non_dif,
                                                                   ignore_eval=True)
        # def to_full_vector(predicted_dif):
        #     new_test_vector = dif_classification.copy().astype('float64')
        #     new_test_vector[np.where(dif_classification == 0)] = predicted_zero_one
        #     new_test_vector[np.where(dif_classification == 1)] = predicted_dif
        #     return new_test_vector

        dif_prediction = self.reg.train_eval(train_index_dif, test_index_dif, ignore_eval=True)
        full_prediction = np.empty((dif_classification.size, dif_prediction.shape[1]))
        full_prediction[:] = np.NaN

        full_prediction[np.where(dif_classification == 0)] = \
            predicted_zero_one.repeat(full_prediction.shape[1]).reshape((-1, full_prediction.shape[1]))
        full_prediction[np.where(dif_classification == 1)] = dif_prediction

        assert not np.isnan(full_prediction).any()

        return self.reg.evaluate_regression(full_prediction, test_index)
        
    def combined_eval(self, evaluations):
        return self.reg.combined_eval(evaluations)


class ModelGenerator:
    def __init__(self, model_class, target_names, **model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.target_names = target_names

    def create_model(self, source, target, is_test=False) -> LearningModel:
        model_params_new = self.model_params.copy()
        if is_test:
            if 'fname' in model_params_new:
                model_params_new['fname'] = f"{self.model_params['fname']}_best_test"
            else:
                model_params_new['out_name'] = f"{self.model_params['out_name']}_best_test"

        return self.model_class(target=target, source=source, target_names=self.target_names, **model_params_new)


class NestedCrossValidation:
    def __init__(self, target, source, model_class, models_params, out_name, logger, target_names,
                 inner_splits=5, outer_splits=5, single_test=True):
        self.target = target
        self.source = source
        self.model_class = model_class
        self.model_generators = [ModelGenerator(model_class, target_names, **params) for params in models_params]
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.out_name = out_name
        self.logger = logger
        self.kf = KFold(n_splits=outer_splits)
        self.single_test = single_test

    def run(self):
        test_eval_of_best_models = []
        for test_count, (train_validate_index, test_index) in enumerate(self.kf.split(self.source)):
            self.logger.info(f"outer test {test_count+1}/{self.outer_splits}")
            scores = []
            for model_count, model_generator in enumerate(self.model_generators):
                self.logger.info(f"model {model_count+1}/{len(self.model_generators)}")
                model = model_generator.create_model(source=self.source[train_validate_index],
                                                     target=self.target[train_validate_index])
                if self.single_test:
                    score = run_single_test(self.inner_splits, model, self.logger)
                else:
                    score = run_kfold_test(self.inner_splits, model, self.logger)

                self.logger.info(f"configuration : {model.out_name}, score: {score}")
                scores.append(score)

            best_model_generator = self.model_generators[np.argmax(scores)]
            best_model_test = best_model_generator.create_model(source=self.source, target=self.target, is_test=True)
            self.logger.info(f"best configuration: {best_model_test.out_name}")

            test_eval_of_best_models.append(best_model_test.train_eval(train_validate_index, test_index))

        count = 1
        new_fname = self.out_name
        while os.path.isdir(new_fname):
            new_fname = self.out_name + f"_({count})"
            count += 1
        self.out_name = new_fname

        os.mkdir(self.out_name)
        best_model_test.set_output(self.out_name)
        best_model_test.combined_eval(test_eval_of_best_models)


def run_kfold_test(n_splits, model, logger):
    kf_test = KFoldTest(n_splits, model, logger)
    kf_test.run_test()
    results = kf_test.evaluate()
    logger.info(results)
    return results


def run_single_test(n_splits, model, logger):
    kf = KFold(n_splits=n_splits)
    train_index, test_index = next(kf.split(model.source))
    score = model.train_eval(train_index, test_index)
    results = model.combined_eval([score])
    logger.info(results)
    return results


def get_binary_evaluation(actual, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(actual, prediction).ravel()
    auc = roc_auc_score(actual, probability)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'AUC-ROC': auc,
            'precision': precision_score(actual, prediction),
            'recall': recall_score(actual, prediction),
            'accuracy': accuracy_score(actual, prediction)
            }


def get_binary_evaluation_many_targets(actual, prediction, probability, class_names):
    results_df = pd.DataFrame(columns=['TP', 'TN', 'FP', 'AUC-ROC', 'precision', 'recall', 'accuracy'])
    for actual_vec, pred_vec, prob_vec, name in zip(actual.T, prediction.T, probability.T, class_names):
        result = get_binary_evaluation(actual_vec, pred_vec, prob_vec)
        results_df = results_df.append(pd.Series(result, name=name))

    results_df.loc['average'] = results_df.mean()
    return results_df


def to_differentiality_label(target):
    std = target.std(axis=1)

    label = std.copy()
    label[std > 0.1] = 1
    label[std < 0.1] = 0

    return label


def to_zero_one(target):
    mean = target.mean(axis=1)
    std = target.std(axis=1)

    label = std.copy()
    label[(std < 0.1) & (mean > 0.5)] = 1
    label[(std < 0.1) & (mean < 0.5)] = 0
    label[std > 0.1] = np.NaN

    return label

