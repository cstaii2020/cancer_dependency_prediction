import time
import pandas as pd

from sklearn.cross_decomposition import CCA
import numpy as np
from scipy import stats, linalg
import json
import pickle

from data_handler import get_new_gene_cancer_data_differentiable_q4, load_new_drug_mean_dependency, \
    get_new_gene_cancer_data_q4, get_cell_line_prediction_differential_genes_q4, \
    get_mean_dependency_to_drug_sensitivity, load_new_drug_mean_expression, load_new_drug_from_exp_dep
from models.cca import tutorial_on_cca, cca_gamma, evaluate_cca


def column_correlation(x, y, n_col):
    return np.diag(np.corrcoef(x, y, rowvar=False)[:n_col, n_col:])


class CCAWrapper(CCA):
    def predict(self, X, copy=True):
        # self.x_rotations_
        return np.dot(self.transform(X), self.y_loadings_.T)


def get_cca_gamma(X, Y, n_comp=10):
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    # normalize/scale
    X -= X.mean(axis=0)
    # X /= cca.x_std_
    Y -= Y.mean(axis=0)
    # Y /= cca.y_std_
    x_std = X.std(axis=0, ddof=1)
    x_std[x_std == 0.0] = 1.0
    X /= x_std
    y_std = Y.std(axis=0, ddof=1)
    y_std[y_std == 0.0] = 1.0
    Y /= y_std

    # rho, _, _ = cca_gamma(X, Y)
    rho, pvalue = evaluate_cca(X, Y, gamma=False)
    # with open("rho_drug_exp.pickle", "wb") as f:
    #     pickle.dump(rho, f)

    print("rho")
    print(json.dumps(dict(enumerate(rho)), indent=2))


def get_cca_from_tutorial(X, Y, n_comp=10):
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    rho, wa, wb = tutorial_on_cca(X, Y)

    print("rho")
    print(json.dumps(dict(enumerate(rho)), indent=2))

    # normalize/scale
    X -= X.mean(axis=0)
    # X /= cca.x_std_
    Y -= Y.mean(axis=0)
    # Y /= cca.y_std_
    x_std = X.std(axis=0, ddof=1)
    x_std[x_std == 0.0] = 1.0
    X /= x_std
    y_std = Y.std(axis=0, ddof=1)
    y_std[y_std == 0.0] = 1.0
    Y /= y_std

    x_scores = X @ wa[:, :n_comp]
    y_scores = Y @ wb[:, :n_comp]

    y_predicted = X @ wa @ np.linalg.pinv(wb)
    print(np.diag(np.corrcoef(Y, y_predicted, rowvar=False)[:n_comp, n_comp:]))

    # print("x_scores.shape", x_scores.shape)
    # print("y_scores.shape", y_scores.shape)

    # correlations = np.diag(np.corrcoef(x_scores, y_scores, rowvar=False)[:n_comp, n_comp:])
    # print(correlations)

    return x_scores, y_scores


def get_cca(X, Y, n_comp=10):
    cca = CCA(n_components=n_comp)
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    x_scores, y_scores = cca.fit_transform(X, Y)

    # Manual Transform
    X -= cca.x_mean_
    X /= cca.x_std_
    Y -= cca.y_mean_
    Y /= cca.y_std_
    calc_scores_x = np.dot(X, cca.x_rotations_)
    calc_scores_y = np.dot(Y, cca.y_rotations_)
    # id_x = cca.x_rotations_ @ linalg.pinv2(cca.x_rotations_)
    # id_y = cca.y_rotations_ @ linalg.pinv2(cca.y_rotations_)

    print("x_scores.shape", x_scores.shape)
    print("y_scores.shape", y_scores.shape)

    correlations = np.diag(np.corrcoef(x_scores, y_scores, rowvar=False)[:n_comp, n_comp:])
    calc_correlations = np.diag(np.corrcoef(calc_scores_x, calc_scores_y, rowvar=False)[:n_comp, n_comp:])

    print(correlations)
    print(calc_correlations)
    return x_scores, y_scores


def cca(func, n_comp=10):
    target, source = func()
    x_scores_tuto, y_scores_tuto = get_cca_from_tutorial(source.values, target.values, n_comp)

    # x_scores_sci, y_scores_sci = get_cca(source.values, target.values, n_comp)
    # get_cca_gamma(source.values, target.values)
    # print(x_scores_sci - x_scores_tuto)


def get_proteomics_data():
    clinical = pd.read_excel(r"C:\Users\Nitay\Downloads\Clinical data for CCA_updated.xlsx")
    proteomics = pd.read_csv(r"C:\Users\Nitay\Downloads\proteomic data_TUMOR_with UNIPROT and ENTREZ IDs.txt", sep="\t")
    clinical = clinical.set_index("Names_new")
    proteomics = proteomics.T
    clinical.drop(['Cellularity (per 1 mm2)'], axis=1, inplace=True)
    samples_index = clinical.index
    clinical_data = clinical.loc[samples_index].iloc[:, 1:].values
    proteomics_data = proteomics.loc[samples_index].values
    return clinical_data.astype('float64'), proteomics_data.astype('float64')


if __name__ == "__main__":
    start = time.time()
    Y, X = get_proteomics_data()
    get_cca(X, Y, n_comp=5)
    # get_cca_gamma(X, Y, n_comp=5)
    # cca(load_new_drug_mean_dependency, n_comp=50)
    # drug, dep = load_new_drug_mean_dependency()
    # print(drug.shape)
    # print(dep.shape)
    print(f"{time.time() - start} seconds")