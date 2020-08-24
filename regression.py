import pickle
import time

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from data_handler import get_cell_line_prediction_differential_genes_q4, \
    load_propagated_cancer_mutations_data_diff_dependency, load_drug_sens_expression, \
    get_new_gene_cancer_data_differentiable_q4, load_targeted_drug_sens_expression, load_targeted_drug_sens_dependency, \
    get_mean_dependency_to_drug_sensitivity, load_new_drug_mean_expression, load_one_target_expression_drug_sens, \
    load_new_drug_mean_dependency, get_new_gene_cancer_data_q4, load_new_drug_from_exp_dep
from logger import setup_logger
from models.base_models import run_kfold_test, NestedCrossValidation, Regressor, CombinedRegressor, Classifier, \
    to_differentiality_label, to_zero_one, IdentityModel
from models.neural_net_models import NeuralNetRegressor, OneToOneDeepNet, NeuralNetClassifier
from sklearn.linear_model import LinearRegression, Ridge


def nn_regression(load_func, n_splits=5, is_clf=False, **kwargs):
    target, source = load_func()
    source_values = source.values
    target_values = target.values
    # if not is_linear:
    if is_clf:
        model = NeuralNetClassifier(source=source_values, target=target_values, target_names=list(target.columns),
                                    **kwargs)
    else:
        model = NeuralNetRegressor(source=source_values, target=target_values, target_names=list(target.columns),
                                   **kwargs)

    run_kfold_test(n_splits, model, logger)


def nn_regression_per_column(load_func, fname, n_splits=5, **kwargs):
    target, source = load_func()
    for i, c in enumerate(target.columns):
        logger.info(f"column : {c} | {i} / {len(target.columns)}")
        curr_source = source.loc[:, [c]].values
        curr_target = target.loc[:, [c]].values
        curr_fname = f"{fname}_{c.split(':')[0]}"
        model = NeuralNetRegressor(source=curr_source, target=curr_target, target_names=[c],
                                   fname=curr_fname, **kwargs)
        run_kfold_test(n_splits, model, logger)


def regression_tests():
    # nn_regression(load_func=get_cell_line_prediction_differential_genes,
    #               hidden_layers=[],
    #               epochs=range(10, 101, 10),
    #               logger=logger,
    #               fname="output/regression/new_cell_line/linear/test1",
    #               is_linear=True)

    # [[10, 10], [10, 10, 10, 10], [100], [500], [100, 100], [500, 500], [100, 100, 100, 100]]:
    for hd in [[10, 10], [100, 100], [500, 500], [1000, 1000, 1000], [100, 100, 100]]:
        # nn_regression(load_func=get_differential_cancer_data,
        #               hidden_layers=hd,
        #               epochs=range(10, 101, 10),
        #               logger=logger,
        #               fname="output/regression/new_gene")

        nn_regression(load_func=get_cell_line_prediction_differential_genes_q4,
                      hidden_layers=hd,
                      epochs=range(10, 101, 10),
                      logger=logger,
                      fname="output/regression/new_cell_line/nn/test1")


def regression_with_autoencoders(hd=[1000] * 5, epochs=range(10, 101, 10), source_only=False, is_linear=False,
                                 file_name="test_500_encoding", **kwargs):
    logger.info("autoencoder test")
    # with open("input/source_new_500_q4_erange(10, 101, 10)_b32_l0.0001_d[2000, 500]_wd_0.0/model.pickle", "rb") as f:
    with open("input/expression_dependency_source_erange(10, 201, 10)_b32_l0.0001_d[500, 500]_wd_0.0/model.pickle",
              "rb") as f:
        source_autoencoder = pickle.load(f)

    # with open("input/target_new_500_q4_erange(10, 101, 10)_b32_l0.0001_d[2000, 500]_wd_0.0/model.pickle", "rb") as f:
    with open("input/expression_dependency_target_erange(10, 201, 10)_b32_l0.0001_d[300, 300]_wd_0.0/model.pickle",
              "rb") as f:
        target_autoencoder = pickle.load(f)

    if not source_only:
        nn_regression(load_func=get_cell_line_prediction_differential_genes_q4,
                      hidden_layers=hd,
                      epochs=epochs,
                      logger=logger,
                      fname=f"output4/regression/new_cell_line/{'linear' if is_linear else 'nn'}/{file_name}",
                      source_autoencoder=source_autoencoder, target_autoencoder=target_autoencoder,
                      is_linear=is_linear,
                      **kwargs)

    else:
        nn_regression(load_func=get_cell_line_prediction_differential_genes_q4,
                      hidden_layers=hd,
                      epochs=epochs,
                      logger=logger,
                      fname=f"output4/regression/new_cell_line/{'linear' if is_linear else 'nn'}/{file_name}",
                      source_autoencoder=source_autoencoder,
                      is_linear=is_linear,
                      **kwargs)


def learn_from_propagated_mutations(hd=[1000, 1000]):
    logger.info("from mutations")
    nn_regression(load_func=load_propagated_cancer_mutations_data_diff_dependency,
                  hidden_layers=hd,
                  epochs=range(10, 201, 10),
                  logger=logger,
                  fname="output/regression/new_cell_line/nn/propagated_mutations")


def test_nn_regression(load_func, folder, name, is_linear, hd, epochs, **kwargs):
    linearity_str = 'linear' if is_linear else 'nn'
    fname = f"output4/regression/{folder}/{linearity_str}/{name}"
    for k, v in kwargs.items():
        fname = f"{fname}_{k}_{v}"
    logger.info(f"{fname}")

    nn_regression(load_func=load_func,
                  hidden_layers=hd,
                  epochs=epochs,
                  logger=logger,
                  fname=fname,
                  is_linear=is_linear,
                  **kwargs)


def from_expression(is_linear=False, hd=[1000, 1000], epochs=range(10, 101, 10), lr=1e-4,
                    file_name="expression_true_combination", **kwargs):
    linearity_str = 'linear' if is_linear else 'nn'
    logger.info(f"from expression {linearity_str}")
    fname = f"output4/regression/new_cell_line/{linearity_str}/{file_name}"
    for k, v in kwargs.items():
        fname = f"{fname}_{k}_{v}"
    nn_regression(load_func=get_cell_line_prediction_differential_genes_q4,
                  hidden_layers=hd,
                  epochs=epochs,
                  logger=logger,
                  lr=lr,
                  fname=fname,
                  is_linear=is_linear,
                  **kwargs)


def expression_to_drug_sensitivity(is_linear=False, hd=[1000, 1000], epochs=range(10, 101, 10), targeted=True):
    linearity_str = 'linear' if is_linear else 'nn'

    logger.info(f"drug sensitivity {linearity_str} {'only targeted' if targeted else ''}")

    nn_regression(load_func=load_targeted_drug_sens_expression if targeted else load_drug_sens_expression,
                  hidden_layers=hd,
                  epochs=epochs,
                  logger=logger,
                  fname=f"output4/regression/drug/{linearity_str}/expression_drug_sensitivity{'_targeted' if targeted else ''}",
                  is_linear=is_linear)


def dependency_to_drug_sens(is_linear=False, hd=[1000, 1000], epochs=range(10, 101, 10)):
    linearity_str = 'linear' if is_linear else 'nn'

    nn_regression(load_func=load_targeted_drug_sens_dependency,
                  hidden_layers=hd,
                  epochs=epochs,
                  logger=logger,
                  fname=f"output4/regression/drug/{linearity_str}/dependency_to_drug",
                  is_linear=is_linear)


def test_new_drug_mean_expression(is_linear=False, hd=[1000, 1000], epochs=range(10, 101, 10), **kwargs):
    test_nn_regression(load_func=load_new_drug_mean_expression, folder='drug', name='new_drug_exp',
                       is_linear=is_linear, hd=hd, epochs=epochs, **kwargs)


def test_new_drug_mean_dependency(is_linear=False, hd=[1000, 1000], epochs=range(10, 101, 10), **kwargs):
    test_nn_regression(load_func=load_new_drug_mean_dependency, folder='drug', name='new_drug_mean_dep',
                       is_linear=is_linear, hd=hd, epochs=epochs, **kwargs)


def test_new_drug_from_dep_exp(hd=[1000, 1000], epochs=range(10, 101, 10), is_linear=False, **kwargs):
    test_nn_regression(load_func=load_new_drug_from_exp_dep, folder='drug', name='new_drug_concatenation',
                       is_linear=is_linear, hd=hd, epochs=epochs, **kwargs)



# def mean_dependency_to_drug_sensitivity(is_linear=False, hd=[1000, 1000], epochs=range(10, 101, 10), **kwargs):
#     linearity_str = 'linear' if is_linear else 'nn'
#
#     nn_regression(load_func=get_mean_dependency_to_drug_sensitivity,
#                   fname=f"output4/regression/drug/{linearity_str}/mean_dep_one_by_one",
#                   hidden_layers=hd,
#                   epochs=epochs,
#                   logger=logger,
#                   is_linear=is_linear,
#                   **kwargs)


def new_gene_diff_from_expression(is_linear=False, hd=[1000, 1000], epochs=range(10, 121, 10), **kwargs):
    linearity_str = 'linear' if is_linear else 'nn'
    logger.info(f"new gene from expression {linearity_str}")
    fname = f"output4/regression/new_gene/{linearity_str}/expression_differential"
    for k, v in kwargs.items():
        fname = f"{fname}_{k}_{v}"
    nn_regression(load_func=get_new_gene_cancer_data_differentiable_q4,
                  hidden_layers=hd,
                  epochs=epochs,
                  logger=logger,
                  fname=fname,
                  is_linear=is_linear,
                  **kwargs)


def new_all_gene_from_expression(is_linear=False, hd=[1000, 1000], epochs=range(10, 121, 10), **kwargs):
    linearity_str = 'linear' if is_linear else 'nn'
    logger.info(f"new gene from expression {linearity_str}")
    fname = f"output4/regression/new_gene/{linearity_str}/expression_all_genes"
    for k, v in kwargs.items():
        fname = f"{fname}_{k}_{v}"

    nn_regression(load_func=get_new_gene_cancer_data_q4,
                  hidden_layers=hd,
                  epochs=epochs,
                  logger=logger,
                  fname=fname,
                  is_linear=is_linear,
                  **kwargs)


def run_nested(load_func, params, out_name, inner_split=5, outer_split=5, model_class=NeuralNetRegressor):
    target, source = load_func()
    NestedCrossValidation(target=target.values,
                          source=source.values,
                          model_class=model_class,
                          models_params=params,
                          out_name=out_name,
                          logger=logger,
                          inner_splits=inner_split,
                          outer_splits=outer_split,
                          target_names=target.columns).run()


def nested_from_expression(autoencoder=False):
    if autoencoder:
        autoencoder_str = "encoding"
        logger.info("autoencoder test")
        with open("input/source_new_500_q4_erange(10, 101, 10)_b32_l0.0001_d[2000, 500]_wd_0.0/model.pickle",
                  "rb") as f:
            source_autoencoder = pickle.load(f)

        with open("input/target_new_500_q4_erange(10, 101, 10)_b32_l0.0001_d[2000, 500]_wd_0.0/model.pickle",
                  "rb") as f:
            target_autoencoder = pickle.load(f)
    else:
        logger.info("no encoding")
        autoencoder_str = "no_encoding"

    name = f"output4/nested/new_cell_line/{autoencoder_str}/test"
    params = [{
        'hidden_layers': [1000] * 8,
        'epochs': range(10, 201, 10),
        'logger': logger,
        'fname': name,
    }]

    if autoencoder:
        for p in params:
            p['target_autoencoder'] = target_autoencoder
            p['source_autoencoder'] = source_autoencoder

    run_nested(load_func=get_cell_line_prediction_differential_genes_q4,
               params=params, out_name=f"{name}_combination")


def nested_new_gene_all(name):
        name = f"output4/nested/new_gene/all_{name}"
        params = []
        hidden_layers = [[50, 50], [100, 100], [200, 200], [500, 500]]
        all_comp = [10, 50, 100, 200]
        for hd in hidden_layers:
            for c in all_comp:
                params.append({
                    'hidden_layers': hd,
                    'epochs': range(10, 151, 10),
                    'logger': logger,
                    'fname': name,
                    'comp': c})

        # params = [{
        #     'hidden_layers': [100] * 1,
        #     'epochs': range(10, 201, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [100] * 2,
        #     'epochs': range(10, 201, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [100] * 3,
        #     'epochs': range(10, 201, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [100] * 5,
        #     'epochs': range(10, 201, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [500] * 1,
        #     'epochs': range(10, 101, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [500] * 2,
        #     'epochs': range(10, 101, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [500] * 3,
        #     'epochs': range(10, 101, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }, {
        #     'hidden_layers': [500] * 5,
        #     'epochs': range(10, 101, 10),
        #     'logger': logger,
        #     'fname': name,
        #     'comp': 50,
        #     'use_scikit': 'pca'
        # }
        # ]

        run_nested(load_func=get_new_gene_cancer_data_q4,
                   params=params, out_name=f"{name}_combination")


def nested_new_gene_all_linear_ridge(name):
    name = f"output4/nested/new_gene/linear_ridge/all_{name}"
    params = [{
        'reg': LinearRegression(),
        'out_name': name
    }, {
        'reg': Ridge(),
        'out_name': name
    }, {
        'reg': Ridge(alpha=10),
        'out_name': name
    }, {
        'reg': Ridge(alpha=100),
        'out_name': name
    }, {
        'reg': Ridge(alpha=1000),
        'out_name': name
    }, {
        'reg': Ridge(alpha=10000),
        'out_name': name
    }
    ]

    run_nested(load_func=get_new_gene_cancer_data_q4,
               params=params, out_name=f"{name}_combination", model_class=Regressor)


def nested_drug_ridge(name, func=load_new_drug_from_exp_dep):
    fname = f"output4/nested/drug/linear_ridge/{name}"
    params = []
    for log_alpha in [None, 0, 1, 2, 3, 4]:
        for comp in [100, 200, 500]:
            if log_alpha is None:
                reg = LinearRegression()
            else:
                reg = Ridge(alpha=10**log_alpha)
            params.append({
                'reg': reg,
                'out_name': fname,
                'comp': comp,
                'use_scikit': 'pca'
            })
    # # params = [{
    # #     'reg': LinearRegression(),
    # #     'out_name': name
    # # }, {
    # #     'reg': Ridge(),
    # #     'out_name': name
    # # }, {
    # #     'reg': Ridge(alpha=10),
    # #     'out_name': name
    # # }, {
    # #     'reg': Ridge(alpha=100),
    # #     'out_name': name
    # # }, {
    # #     'reg': Ridge(alpha=1000),
    # #     'out_name': name
    # # }, {
    # #     'reg': Ridge(alpha=10000),
    # #     'out_name': name
    # # }
    # ]

    run_nested(load_func=func,
               params=params, out_name=f"{fname}_combination", model_class=Regressor)


def test_nested_drug_from_dep_exp(name, func=load_new_drug_from_exp_dep):
    fname = f"output4/nested/drug/nn/{name}"
    params = []
    for hd in [[100], [200], [100]*2, [200]*2, [500], [500]*2]:
        for comp in [100, 200]:
            params.append({
                'hidden_layers': hd,
                'epochs': range(10, 201, 10),
                'logger': logger,
                'fname': fname,
                'comp': comp,
                'use_scikit': 'pca'
            })

    run_nested(load_func=func,
               params=params, out_name=f"{fname}_combination")


def regressor_test(func, folder, reg=LinearRegression(), name="linear", **kwargs):
    target, source = func()
    regressor = Regressor(target=target.values, source=source.values, target_names=target.columns,
                          out_name=f"output4/regression/{folder}/scikit_learn/{name}",
                          reg=reg, **kwargs)
    start = time.time()
    run_kfold_test(5, regressor, logger)
    logger.info(f"CV of model took {time.time() - start} seconds")


def new_gene_regressor(reg=LinearRegression(), name="linear", **kwargs):
    regressor_test(get_new_gene_cancer_data_differentiable_q4, "new_gene", reg=reg, name=name, **kwargs)


def new_cell_line_regressor(reg=LinearRegression(), name="linear", **kwargs):
    regressor_test(get_cell_line_prediction_differential_genes_q4, "new_cell_line", reg=reg, name=name, **kwargs)


def new_drug_mean_dep_regressor(reg=LinearRegression(), name="linear", **kwargs):
    regressor_test(load_new_drug_mean_dependency, "drug", reg=reg, name=name, **kwargs)


def new_drug_mean_exp_regressor(reg=LinearRegression(), name="linear", **kwargs):
    regressor_test(load_new_drug_mean_expression, "drug", reg=reg, name=f"expression_{name}", **kwargs)


def combined_regression(scikit_reg=Ridge(alpha=1000), **kwargs):
    out_name = "output4/regression/combined/test"
    target, source = get_new_gene_cancer_data_q4()
    target_values, source_values = target.values, source.values

    random_forest_clf_dif = RandomForestClassifier(n_estimators=20)
    random_forest_clf_zero = RandomForestClassifier(n_estimators=20)
    dif_classifier = Classifier(source_values, to_differentiality_label(target_values),
                                random_forest_clf_dif, [0, 1], f"{out_name}_diff")
    zero_classifier = Classifier(source_values, to_zero_one(target_values),
                                 random_forest_clf_zero, [0, 1], f"{out_name}_zero")

    # reg = NeuralNetRegressor(source=source_values, target=target_values, logger=logger,
    #                          fname=f"{out_name}_nn", target_names=list(target.columns), **kwargs)
    reg = Regressor(target=target.values, source=source.values, target_names=list(target.columns),
                    out_name=f"{out_name}_nn",
                    reg=scikit_reg, **kwargs)

    combined_reg = CombinedRegressor(source=source, target=target,
                                     out_name=f"{out_name}_combined",
                                     dif_classifier=dif_classifier,
                                     zero_classifier=zero_classifier,
                                     reg=reg)

    run_kfold_test(5, combined_reg, logger)


if __name__ == "__main__":
    logger = setup_logger("regression")
    # test_nested_drug_from_dep_exp("second")
    # nested_drug_ridge("combined")
    # nested_drug_ridge("dep", func=load_new_drug_mean_dependency)
    # nested_drug_ridge("exp", func=load_new_drug_mean_expression)
    # test_nested_drug_from_dep_exp("expression", load_drug_sens_expression)
    # test_nested_drug_from_dep_exp("dependency", load_new_drug_mean_dependency)
    # nested_new_gene_all('different_comp')
    # test_new_drug_from_dep_exp(hd=[500], wd=1e-4, epochs=range(10, 201, 10), comp=200, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[200], wd=1e-4, epochs=range(10, 201, 10), comp=200, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[500], epochs=range(10, 201, 10), comp=200, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[100], epochs=range(10, 201, 10), comp=100, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[200], epochs=range(10, 201, 10), comp=100, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[200]*2, epochs=range(10, 201, 10), comp=200, use_scikit='pca', wd=1e-2)
    # test_new_drug_from_dep_exp(hd=[500], wd=1e-3, epochs=range(10, 201, 10), comp=500, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[500], wd=1e-4, epochs=range(10, 201, 10), comp=500, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[500], epochs=range(10, 201, 10), comp=500, use_scikit='pca')
    # test_new_drug_from_dep_exp(hd=[500], wd=1e-4, epochs=range(10, 201, 10))
    # test_new_drug_from_dep_exp(hd=[500], epochs=range(10, 201, 10))
    # test_new_drug_mean_dependency(hd=[500], epochs=range(10, 151, 10))

    regressor_test(load_new_drug_from_exp_dep, 'drug', reg=IdentityModel(), name="identity")
    # new_drug_mean_dependency_test(comp=200, hd=[500]*2, epochs=range(10, 201, 10), use_scikit="pca")
    # new_drug_mean_dep_regressor(reg=Ridge(alpha=1000), comp=200, use_scikit='pca', name="ridge_1000_pca_50")
    # from_expression(hd=[500], epochs=range(10, 10001, 10), wd=1e-5, comp=500, use_scikit='pca')
    # new_cell_line_regressor(reg=Ridge(alpha=1000), comp=500, use_scikit='pca')
    # regressor_test(get_new_gene_cancer_data_q4, "new_gene",
    #                reg=Ridge(alpha=1000), name="ridge_1000_cca_50", comp=50)
    # from_expression(hd=[1000]*8, epochs=range(10, 301, 10), wd=1e-2)
    # from_expression(hd=[1000]*8, epochs=range(10, 301, 10), wd=1e-3)
    # from_expression(hd=[1000]*8, epochs=range(10, 301, 10), wd=1e-4)
    # from_expression(hd=[400]*12, epochs=range(10, 301, 10), wd=1e-2, comp=500, use_scikit='pca')
    # from_expression(hd=[400]*12, epochs=range(10, 301, 10), wd=1e-3, comp=500, use_scikit='pca')
    # from_expression(hd=[400]*12, epochs=range(10, 301, 10), wd=1e-4, comp=500, use_scikit='pca')
    # new_cell_line_regressor(reg=Ridge(alpha=10))
    # new_cell_line_regressor(reg=Ridge(alpha=100))
    # new_cell_line_regressor(reg=Ridge(alpha=10000, solver='saga'), name="ridge_10p4")
    # regressor_test(get_new_gene_cancer_data_q4, "new_gene", name="linear_all_genes_ridge_0.1", reg=Ridge(alpha=0.1))
    # combined_regression()
    # nested_new_gene_all("all_gene_first_cca_test")
    # regressor_test(get_new_gene_cancer_data_q4, "new_gene", reg=Ridge(), name="all_genes_ridge")
    # regressor_test(get_new_gene_cancer_data_q4, "new_gene", reg=Ridge(alpha=10000),
    #                name="all_genes_ridge_10000_cca")
    # new_all_gene_from_expression(hd=[200]*2, epochs=range(10, 101, 10), comp=200)
    # new_all_gene_from_expression(hd=[500]*2, epochs=range(10, 101, 10), comp=500)
    # new_all_gene_from_expression(hd=[100]*5, epochs=range(10, 201, 10), comp=100)
    # new_all_gene_from_expression(hd=[100]*2, epochs=range(10, 101, 10), comp=50)
    # new_all_gene_from_expression(hd=[500]*2, epochs=range(10, 101, 10), comp=50)
    # new_gene_diff_from_expression(hd=[500]*2, epochs=range(10, 101, 10), comp=200)
    # new_all_gene_from_expression(hd=[100]*2, epochs=range(10, 301, 10), comp=5)
    # new_gene_regressor(reg=MLPRegressor(hidden_layer_sizes=(1000, 1000), activation='tanh', alpha=0),
    #                    name="NN_[1000,1000]_tanh_wd_0")
    # new_gene_regressor(reg=MLPRegressor(hidden_layer_sizes=(500, 500), activation='tanh'),
    #                    name="NN_[500,500]_tanh_wd_.0001")
    # new_gene_regressor(reg=MLPRegressor(hidden_layer_sizes=(630,), activation='relu'),
    #                    name="NN_[200,200]_relu")
    # from_expression(hd=[400]*15, comp=500, use_scikit='pca', epochs=range(10, 401, 10))
    # mean_dependency_to_drug_sensitivity(comp=260, epochs=range(10, 201, 10))
    # new_all_gene_from_expression(hd=[500] * 5, epochs=range(10, 151, 10), comp=50, use_scikit='pca')
    # new_all_gene_from_expression(hd=[500] * 6, epochs=range(10, 151, 10), comp=50, use_scikit='pca')
    # new_all_gene_from_expression(hd=[500] * 7, epochs=range(10, 151, 10), comp=50, use_scikit='pca')
    # nested_new_gene_all("pca/first")
    # nested_new_gene_all_linear_ridge("first")
    # new_all_gene_from_expression(hd=[1000] * 3, epochs=range(5, 51, 5))
