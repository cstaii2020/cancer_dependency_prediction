import torch
from scipy.stats import pearsonr
import pandas as pd
import logger as lg
from data_handler import get_cell_line_prediction_differential_genes_q4, load_drug_sens_expression, get_tumor_file, \
    normalize
from models.autoencoders import AutoEncoderModel
from models.base_models import run_kfold_test, KFoldTest
import pickle

from models.cca_learning import CcaClassifier


class AutoEncoderTrailExecutor:
    def __init__(self, load_func, source_fname, target_fname, logger, hd=[2000, 500],
                 epochs_source=range(10, 101, 10), epochs_target=range(10, 101, 10)):
        logger.info(f"hd is {hd}")
        self.logger = logger

        target, source = load_func()
        self.source_encoder = AutoEncoderModel(source=source.values, target_names=list(source.columns),
                                               hidden_layers=hd, epochs=epochs_source, logger=logger,
                                               fname=source_fname)

        self.target_encoder = AutoEncoderModel(source=target.values, target_names=list(target.columns),
                                               hidden_layers=hd, epochs=epochs_target, logger=logger,
                                               fname=target_fname)

    def run_source_test(self, n_splits=5):
        self.logger.info("source encoder test")
        run_kfold_test(n_splits, self.source_encoder, self.logger)

    def run_target_test(self, n_splits=5):
        self.logger.info("target encoder test")
        run_kfold_test(n_splits, self.target_encoder, self.logger)

    def run_test(self, n_splits=5):
        self.run_source_test(n_splits)
        self.run_target_test(n_splits)

    def save_source_encoder(self):
        self.logger.info("save source encoder")
        self.source_encoder.train_and_save()

    def save_target_encoder(self):
        self.logger.info("save target encoder")
        self.target_encoder.train_and_save()

    def save_encoders(self):
        self.save_source_encoder()
        self.save_target_encoder()


# def autoencoder_kfold(load_func, n_splits=5, skip_source=False, skip_target=False, **kwargs):
#     target, source = load_func()
#     source_encoder = AutoEncoderModel(source=source.values, target_names=list(source.columns), **kwargs)
#     run_kfold_test(n_splits, source_encoder, logger)
#
#     target_encoder = AutoEncoderModel(source=target.values, target_names=list(target.columns), **kwargs)
#     run_kfold_test(n_splits, target_encoder, logger)
#
#
# def save_encoder(load_func, fname_source, fname_target, **kwargs):
#     target, source = load_func()
#     logger.info("source encoder:")
#     source_encoder = AutoEncoderModel(source=source.values, target_names=list(source.columns),
#                                       fname=fname_source, **kwargs)
#     source_encoder.train_and_save()
#
#     logger.info("target encoder:")
#     target_encoder = AutoEncoderModel(source=target.values, target_names=list(target.columns),
#                                       fname=fname_target, **kwargs)
#     target_encoder.train_and_save()
#
#
# def test_encoders(load_func, source_fname, target_fname,
#                   skip_source=False, skip_target=False, hd=[2000, 500],
#                   epochs_source=range(10, 101, 10), epochs_target=range(10, 101, 10)):
#     if not skip_source:
#         autoencoder_kfold(load_func=load_func,
#                           hidden_layers=hd,
#                           epochs=epochs_source,
#                           logger=logger,
#                           fname=source_fname)
#
#     if not skip_target:
#         autoencoder_kfold(load_func=load_func,
#                           hidden_layers=hd,
#                           epochs=epochs_target,
#                           logger=logger,
#                           fname=target_fname)
#
#
# def test_encoders_dep(hd=[2000, 500], epochs_source=range(10, 101, 10),
#                       epochs_target=range(10, 101, 10), skip_source=False, skip_target=False):
#     test_encoders(get_cell_line_prediction_differential_genes_q4,
#                   "output4/autoencoder/source_dep_encoder_test",
#                   "output4/autoencoder/target_dep_encoder_test",
#                   hd=hd, epochs_source=epochs_source, epochs_target=epochs_target,
#                   skip_source=skip_source, skip_target=skip_target)
#
#
# def test_encoders_drug(hd=[2000, 500], epochs_source=range(10, 101, 10),
#                        epochs_target=range(10, 101, 10), skip_source=False, skip_target=False):
#     test_encoders(load_drug_sens_expression,
#                   "output4/autoencoder/source_drug_encoder_test",
#                   "output4/autoencoder/target_drug_encoder_test",
#                   hd=hd, epochs_source=epochs_source, epochs_target=epochs_target,
#                   skip_source=skip_source, skip_target=skip_target)
#
#
# def cca_between_encoded(file_name):
#     with open("input/source_new_500_q4_erange(10, 101, 10)_b32_l0.0001_d[2000, 500]_wd_0.0/model.pickle", "rb") as f:
#         source_autoencoder = pickle.load(f)
#
#     with open("input/target_new_500_q4_erange(10, 101, 10)_b32_l0.0001_d[2000, 500]_wd_0.0/model.pickle", "rb") as f:
#         target_autoencoder = pickle.load(f)
#
#     target, source = get_cell_line_prediction_differential_genes_q4()
#     encoded_source = source_autoencoder.encode(torch.as_tensor(source.values)).detach().numpy()
#     model = CcaClassifier(encoded_source, target.values, file_name, target_encoder=target_autoencoder,
#                           class_names=target.columns)
#
#     run_kfold_test(n_splits=5, model=model, logger=logger)
#
#
# def save_encoders_program_dep(hd=[2000, 500], epochs=range(10, 101, 10)):
#     save_encoder(load_func=get_cell_line_prediction_differential_genes_q4,
#                  fname_source="output4/autoencoder/source_new_500_q4",
#                  fname_target="output4/autoencoder/target_new_500_q4",
#                  hidden_layers=hd,
#                  epochs=epochs,
#                  logger=logger)
#
#
# def save_encoders_program_drug(hd=[2000, 500], epochs=range(10, 101, 10)):
#     save_encoder(load_func=load_drug_sens_expression,
#                  fname_source="output4/autoencoder/source_new_500_drug",
#                  fname_target="output4/autoencoder/target_new_500_drug",
#                  hidden_layers=hd,
#                  epochs=epochs,
#                  logger=logger)

def get_expression_dependency_executor(**kwargs):
    return AutoEncoderTrailExecutor(
        load_func=get_cell_line_prediction_differential_genes_q4, logger=logger,
        source_fname="output4/autoencoder/new_source/expression_dependency_source",
        target_fname="output4/autoencoder/new_target/expression_dependency_target", **kwargs)


# def get_drug_executor(**kwargs):
#     return AutoEncoderTrailExecutor(
#         load_func=load_drug_sens_expression, logger=logger,
#         source_fname="output4/autoencoder/new_source/expression_drug_source",
#         target_fname="expression_drug_target", **kwargs)


def tumor_reduction(hd, epochs, name):
    data = get_tumor_file()
    model = AutoEncoderModel(source=data.values, target_names=list(data.columns),
                             hidden_layers=hd, epochs=epochs, logger=logger,
                             fname=f"output4/autoencoder/tumor/{name}")
    model.train_and_save()
    # run_kfold_test(5, model, logger)


def tumor_output_reduction():
    data = get_tumor_file()
    with open(r"output4/autoencoder/tumor/second_erange(10, 501, 10)_b32_l0.0001_d[2000, 5]_wd_0.0/model_1.pickle", "rb") as f:
        model = pickle.load(f)

    data_torch = torch.from_numpy(normalize(data.values)[2])
    encoded_data = model.encode(data_torch)
    reconstructed_data = model.decode(encoded_data)
    # reconstructed_data = model(data_torch)
    data_values = data.values
    reconstructed_data = reconstructed_data.detach().numpy()
    cor_avg = 0
    for i in range(data.shape[0]):
        cor_avg += pearsonr(data_values[i], reconstructed_data[i])[0]
    cor_avg /= data.shape[0]
    print(cor_avg)
    print(data.shape[0])

    cor_avg = 0
    for i in range(data.shape[1]):
        cor_avg += pearsonr(data_values[:, i], reconstructed_data[:, i])[0]
    cor_avg /= data.shape[1]
    print(cor_avg)
    print(data.shape[1])

    pd.DataFrame(data=encoded_data.detach().numpy().T, columns=data.index).to_csv(
        "output4/autoencoder/tumor/data/encoded_5.csv")
    # pd.DataFrame(data=reconstructed_data.T, columns=data.index).to_csv(
    #     "output4/autoencoder/tumor/data/reconstructed_3.csv")
    # path = "output4/autoencoder/data/encoded_data_2"
    # pd.DataFrame(data=encoded_data, columns=data.colums)


if __name__ == "__main__":
    logger = lg.setup_logger("autoencoder")
    tumor_output_reduction()
    # expression_executor = get_expression_dependency_executor(epochs_source=range(10, 201, 10), hd=[500, 500])
    # expression_executor.run_source_test()
    # dependency_executor = get_expression_dependency_executor(epochs_target=range(10, 201, 10), hd=[300, 300])
    # dependency_executor.run_target_test()
    # tumor_reduction(hd=[2000, 4], epochs=range(10, 501, 10), name="second")
    # tumor_output_reduction()
    # tumor_reduction(hd=[2000, 3], epochs=range(10, 501, 10), name="second")
    # tumor_reduction(hd=[2000, 1], epochs=range(10, 201, 10), name="second")
    # drug_exe = get_drug_executor()
    # drug_exe.run_test()
    # test_encoders_dep(epochs_target=range(10, 201, 10), skip_source=True)
    # test_encoders()
    # save_encoders_program()
    # cca_between_encoded("output/cca/sklearn500")
