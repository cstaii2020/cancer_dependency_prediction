import random

import pandas as pd
import numpy as np
import os
from functools import reduce
import collections
import pickle

"""
 =================================== locations vs expressions data ================================================
"""

validation_levels = ["Uncertain", "Approved", "Supported", "Validated"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def create_filter_condition(rel, filter_levels):
    return reduce(lambda x, y: x & y, [rel != level for level in filter_levels])


# returns locations and expressions data frames with compatible row order
# 5.7 % of the entries in locations are 1s
def load_location_expression_data(replications=None, filter_levels=None):
    dirname = os.path.dirname(os.path.realpath(__file__))
    locations = pd.read_excel(os.path.join(dirname, "test/locations.xlsx"))
    expressions = pd.read_excel(os.path.join(dirname, "test/expressions.xlsx"))

    locations.set_index("ENSG", inplace=True, drop=True)
    expressions.set_index("ENSG", inplace=True, drop=True)

    # locations = locations.select_dtypes(include=numerics)
    # expressions = expressions.select_dtypes(include=numerics)

    location_col_names = list(locations)
    expression_col_names = list(expressions)

    data = locations.join(expressions, how="inner", lsuffix='_loc', rsuffix='_expr')

    reliability = data["Reliability"]

    if replications:
        data = replicate(data, reliability, replications)

    data = data.select_dtypes(include=numerics)

    locations = data.filter(items=location_col_names)
    expressions = data.filter(items=expression_col_names)

    if filter_levels is not None:
        filter_condition = create_filter_condition(reliability, filter_levels)
        locations = locations[filter_condition]
        expressions = expressions[filter_condition]

    return locations, expressions, reliability


# expression as nparray
# def normalize(expressions):
#     centered_expression = expressions - expressions.mean(axis=1)[:, np.newaxis]
#     return centered_expression / np.linalg.norm(centered_expression, axis=1)[:, np.newaxis]


def get_mean_std(data):
    return data.mean(0), data.std(0)


def normalize(data):
    mean, std = get_mean_std(data)
    return mean, std, normalize_by(mean, std, data)


def normalize_by(mean, std, data):
    return (data - mean) / std


def normalize_by_train(train_indices, data):
    mean, std = get_mean_std(data[train_indices])
    return normalize_by(mean, std, data)


def replicate(data, reliability, replications):
    replicated_data = pd.DataFrame()

    if isinstance(replications, list) and len(replications) == 4:
        for replication, level in zip(replications, validation_levels):
            replica = data[reliability == level]
            if replication != 0:
                replicated_data = replicated_data.append([replica] * replication, ignore_index=True)

    else:
        print("invalid replication input : no replications made")
        return data

    return replicated_data


def replicate_array(data, reliability, replications):
    mapper = pd.Series(replications, validation_levels)
    repeat_by = reliability.map(mapper).values
    return np.repeat(data, repeat_by, axis=0)


"""
================================== cancer depMap data - Achilles ================================
"""

# pickle_path = r"C:\Users\Nitay\PycharmProjects\cca\cancer_data_unfiltered.pickle"
pickle_path = r"cancer_data_dependency.pickle"
# pickle_path = r"C:\Users\Nitay\PycharmProjects\cca\cancer_data_dependency_cca.pickle"


def load_cancer_data():
    if os.path.isfile(pickle_path):
        pickle_load = pickle.load(open(pickle_path, 'rb'))
        return pickle_load[1], pickle_load[0]

    # gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_effect.csv")
    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    expressions_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_expression_full.csv")

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0").T
    expressions_df = expressions_df.set_index("Unnamed: 0").T

    gene_effect_df.columns.names = ["cell_line"]
    expressions_df.columns.names = ["cell_line"]
    gene_effect_df.index.names = ["gene"]
    expressions_df.index.names = ["gene"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(index=clean_gene_name)
    clean_expressions_df = expressions_df.rename(index=clean_gene_name)

    common_genes = set(clean_gene_effect_df.index).intersection(clean_expressions_df.index)

    duplicates = [item for item, count in collections.Counter(list(clean_expressions_df.index)).items() if count > 1]
    fixed_common_genes = common_genes.difference(duplicates)
    duplicate_filtered_expressions_df = clean_expressions_df.loc[~clean_expressions_df.index.duplicated(keep='first')]

    common_cols = set(clean_gene_effect_df.columns).intersection(duplicate_filtered_expressions_df.columns)

    filtered_gene_effect_df = clean_gene_effect_df \
        .filter(items=fixed_common_genes, axis=0) \
        # .filter(items=common_cols, axis=1)
    filtered_expressions_df = duplicate_filtered_expressions_df \
        .filter(items=fixed_common_genes, axis=0) \
        # .filter(items=common_cols, axis=1)

    ## CCA
    # from sklearn.cross_decomposition import CCA
    # Y = filtered_gene_effect_df.values
    # X = filtered_expressions_df.values
    # cca = CCA(n_components=10)
    # cca.fit(X, Y)
    # X_c = cca.transform(X)
    # filtered_expressions_df = pd.DataFrame(X_c)

    # PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=10)
    # X = filtered_expressions_df.values
    # X_c = pca.fit_transform(X)
    # filtered_expressions_df = pd.DataFrame(X_c)

    pickle.dump([filtered_expressions_df, filtered_gene_effect_df], open(pickle_path, 'wb'))

    return filtered_gene_effect_df, filtered_expressions_df


def get_middle(filtered_expressions_df, filtered_gene_effect_df, percentage):
    # part = percentage / 100
    # averages = filtered_gene_effect_df.mean(axis=1)
    # middle_gene_indexes = np.where((averages > part) & (averages < 1 - part))[0]

    # smaller = filtered_gene_effect_df < 0.5 - part
    # smaller_counts = np.count_nonzero(smaller, axis=1)
    # larger = filtered_gene_effect_df > 0.5 + part
    # larger_counts = np.count_nonzero(larger, axis=1)
    # cond1 = (larger_counts / filtered_gene_effect_df.shape[1]) > 0.1
    # cond2 = (smaller_counts / filtered_gene_effect_df.shape[1]) > 0.1
    # middle_gene_indexes = np.where(cond1 & cond2)[0]

    gene_count = filtered_gene_effect_df.shape[0]
    k = int(gene_count * percentage) // 100
    top_std_effect_genes = np.argsort(filtered_gene_effect_df.std(axis=1).values)[-k:]

    middle_filtered_gene_effect_df = filtered_gene_effect_df.iloc[top_std_effect_genes]
    middle_filtered_expressions_df = filtered_expressions_df.iloc[top_std_effect_genes]

    print(f"nunmber of genes filtered to {top_std_effect_genes.size}")
    return middle_filtered_gene_effect_df, middle_filtered_expressions_df


def load_cancer_middle_gene_data(percentage):
    pickle_load = pickle.load(open(pickle_path, 'rb'))
    filtered_gene_effect_df, filtered_expressions_df = pickle_load[1], pickle_load[0]
    # r, c = filtered_expressions_df.shape
    # fake_df = pd.DataFrame(data=np.random.rand(r, c), index=filtered_expressions_df.index,
    #                        columns=filtered_expressions_df.columns)
    return filtered_gene_effect_df, filtered_expressions_df.iloc[:,
                                    [random.randint(0, filtered_expressions_df.shape[1] - 1)]] #fake_df
    # return get_middle(filtered_expressions_df, filtered_gene_effect_df, percentage)


pickle_path_relevant = r"cancer_data_relevant_pca10.pickle"


def load_cancer_data_relevant():
    if os.path.isfile(pickle_path_relevant):
        pickle_load = pickle.load(open(pickle_path_relevant, 'rb'))
        return pickle_load[1], pickle_load[0]

    # gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_effect.csv")
    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    expressions_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_expression_full.csv")

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0").T
    expressions_df = expressions_df.set_index("Unnamed: 0").T

    gene_effect_df.columns.names = ["cell_line"]
    expressions_df.columns.names = ["cell_line"]
    gene_effect_df.index.names = ["gene"]
    expressions_df.index.names = ["gene"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(index=clean_gene_name)
    clean_expressions_df = expressions_df.rename(index=clean_gene_name)

    common_genes = set(clean_gene_effect_df.index).intersection(clean_expressions_df.index)

    duplicates = [item for item, count in collections.Counter(list(clean_expressions_df.index)).items() if count > 1]
    fixed_common_genes = common_genes.difference(duplicates)
    duplicate_filtered_expressions_df = clean_expressions_df.loc[~clean_expressions_df.index.duplicated(keep='first')]

    relevant_genes_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\gene_conversion.txt", sep="\t")

    def get_gene_name(txt):
        return txt.split("(")[-1].replace(")", "")

    get_gene_name_vec = np.vectorize(lambda x: get_gene_name(x))
    relevant_genes = get_gene_name_vec(relevant_genes_df["Gene Name"].values)

    fixed_common_genes = fixed_common_genes.intersection(set(relevant_genes))

    filtered_gene_effect_df = clean_gene_effect_df.filter(items=fixed_common_genes, axis=0)
    filtered_expressions_df = duplicate_filtered_expressions_df.filter(items=fixed_common_genes, axis=0)



    ## CCA
    # from sklearn.cross_decomposition import CCA
    # Y = filtered_gene_effect_df.values
    # X = filtered_expressions_df.values
    # cca = CCA(n_components=10)
    # cca.fit(X, Y)
    # X_c = cca.transform(X)
    # filtered_expressions_df = pd.DataFrame(X_c)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    X = filtered_expressions_df.values
    X_c = pca.fit_transform(X)
    filtered_expressions_df = pd.DataFrame(X_c)

    pickle.dump([filtered_expressions_df, filtered_gene_effect_df], open(pickle_path_relevant, 'wb'))

    return filtered_gene_effect_df, filtered_expressions_df


pickle_path_fake_relevant = r"cancer_data_fake_relevant_pca10.pickle"


def load_cancer_data_fake_relevant():
    if os.path.isfile(pickle_path_fake_relevant):
        pickle_load = pickle.load(open(pickle_path_fake_relevant, 'rb'))
        return pickle_load[1], pickle_load[0]

    # gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_effect.csv")
    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    expressions_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_expression_full.csv")

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0").T
    expressions_df = expressions_df.set_index("Unnamed: 0").T

    gene_effect_df.columns.names = ["cell_line"]
    expressions_df.columns.names = ["cell_line"]
    gene_effect_df.index.names = ["gene"]
    expressions_df.index.names = ["gene"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(index=clean_gene_name)
    clean_expressions_df = expressions_df.rename(index=clean_gene_name)

    common_genes = set(clean_gene_effect_df.index).intersection(clean_expressions_df.index)

    duplicates = [item for item, count in collections.Counter(list(clean_expressions_df.index)).items() if count > 1]
    fixed_common_genes = common_genes.difference(duplicates)
    duplicate_filtered_expressions_df = clean_expressions_df.loc[~clean_expressions_df.index.duplicated(keep='first')]

    import random
    fixed_common_genes = random.sample(fixed_common_genes, k=507)
    # fixed_common_genes = fixed_common_genes.intersection(set(relevant_genes))

    filtered_gene_effect_df = clean_gene_effect_df.filter(items=fixed_common_genes, axis=0)
    filtered_expressions_df = duplicate_filtered_expressions_df.filter(items=fixed_common_genes, axis=0)



    ## CCA
    # from sklearn.cross_decomposition import CCA
    # Y = filtered_gene_effect_df.values
    # X = filtered_expressions_df.values
    # cca = CCA(n_components=10)
    # cca.fit(X, Y)
    # X_c = cca.transform(X)
    # filtered_expressions_df = pd.DataFrame(X_c)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    X = filtered_expressions_df.values
    X_c = pca.fit_transform(X)
    filtered_expressions_df = pd.DataFrame(X_c)

    pickle.dump([filtered_expressions_df, filtered_gene_effect_df], open(pickle_path_fake_relevant, 'wb'))

    return filtered_gene_effect_df, filtered_expressions_df


# pickle_path = r"C:\Users\Nitay\PycharmProjects\cca\cancer_data_unfiltered.pickle"
pickle_group_path = r"cancer_data_dependency_grouped.pickle"
# pickle_path = r"C:\Users\Nitay\PycharmProjects\cca\cancer_data_dependency_cca.pickle"


def load_grouped_cancer_data():
    if os.path.isfile(pickle_group_path):
        pickle_load = pickle.load(open(pickle_group_path, 'rb'))
        return pickle_load[1], pickle_load[0], pickle_load[3], pickle_load[2]

    # gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_effect.csv")
    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    expressions_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_expression_full.csv")

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0").T
    expressions_df = expressions_df.set_index("Unnamed: 0").T

    gene_effect_df.columns.names = ["cell_line"]
    expressions_df.columns.names = ["cell_line"]
    gene_effect_df.index.names = ["gene"]
    expressions_df.index.names = ["gene"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(index=clean_gene_name)
    clean_expressions_df = expressions_df.rename(index=clean_gene_name)

    common_genes = set(clean_gene_effect_df.index).intersection(clean_expressions_df.index)

    duplicates = [item for item, count in collections.Counter(list(clean_expressions_df.index)).items() if count > 1]
    fixed_common_genes = common_genes.difference(duplicates)
    duplicate_filtered_expressions_df = clean_expressions_df.loc[~clean_expressions_df.index.duplicated(keep='first')]

    # common_cols = set(clean_gene_effect_df.columns).intersection(duplicate_filtered_expressions_df.columns)

    filtered_gene_effect_df = clean_gene_effect_df \
        .filter(items=fixed_common_genes, axis=0)
    filtered_expressions_df = duplicate_filtered_expressions_df \
        .filter(items=fixed_common_genes, axis=0)

    cell_lines_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\cell_lines.csv")
    cell_lines_df = cell_lines_df[cell_lines_df['disease'] != " "].set_index("DepMap_ID")

    effect_cell_lines_with_disease = set(filtered_gene_effect_df.columns).intersection(cell_lines_df.index)
    effect_cell_lines_df = cell_lines_df.filter(items=effect_cell_lines_with_disease, axis=0)[["disease"]]
    filtered_gene_effect_df = filtered_gene_effect_df.filter(items=effect_cell_lines_with_disease, axis=1)
    effect_groups = effect_cell_lines_df.reset_index().groupby("disease").groups

    expression_cell_lines_with_disease = set(filtered_expressions_df.columns).intersection(cell_lines_df.index)
    cell_line_expression_df = cell_lines_df.filter(items=expression_cell_lines_with_disease, axis=0)[["disease"]]
    filtered_expressions_df = filtered_expressions_df.filter(items=expression_cell_lines_with_disease, axis=1)
    expression_groups = cell_line_expression_df.reset_index().groupby("disease").groups

    pickle.dump([filtered_expressions_df, filtered_gene_effect_df, expression_groups, effect_groups], open(pickle_group_path, 'wb'))

    return filtered_gene_effect_df, filtered_expressions_df, effect_groups, expression_groups


def load_grouped_cancer_middle_gene_data(percentage):
    pickle_load = pickle.load(open(pickle_group_path, 'rb'))
    filtered_gene_effect_df, filtered_expressions_df = pickle_load[1], pickle_load[0]
    mid_eff, mid_exp = get_middle(filtered_expressions_df, filtered_gene_effect_df, percentage)
    return mid_eff, mid_exp, pickle_load[3], pickle_load[2]


def load_grouped_go_genes_dependency_expression(percentage, group=True):
    target_df, source_df, _, source_group = pickle.load(open("input/grouped_go_genes_dependency_expression.pickle", 'rb'))
    target_df, _ = get_top_std(target_df, source_df, percentage, False)
    #shuffle source genes
    # genes = list(source_df.columns.copy())
    # np.random.shuffle(genes)
    # source_df = source_df[genes]
    if group:
        return target_df, source_df, _, source_group
    else:
        return target_df, source_df


def load_drug_sens_expression():
    [drug_sensitivity_df, expression_df] = pickle.load(open("input/drug_sensitivity_expressions.pickle", "rb"))
    return drug_sensitivity_df, expression_df


def load_new_drug_mean_expression():
    return get_target_source_from_pickle("input/mean_expression_drug_sensitivity.pickle")


def load_one_target_expression_drug_sens():
    return get_target_source_from_pickle("input/one_target_expression_drug_sensitivity.pickle")


def load_targeted_drug_sens_expression():
    [drug_sensitivity_df, expression_df] = pickle.load(open("input/drug_sensitivity_targeted_expressions.pickle", "rb"))
    return drug_sensitivity_df, expression_df


def load_targeted_drug_sens_dependency():
    with open("input/dependency_to_drug_sens.pickle", "rb") as f:
        [drug_sens, dependency] = pickle.load(f)

    return drug_sens, dependency


def get_target_source_from_pickle(path):
    with open(path, "rb") as f:
        [target, source] = pickle.load(f)

    if str(target.values.dtype) == 'object':
        target = target.astype('float64')

    if str(source.values.dtype) == "object":
        source = source.astype('float64')

    return target, source


def get_mean_dependency_to_drug_sensitivity():
    return get_target_source_from_pickle("input/mean_dependency_to_drug.pickle")


def load_new_drug_mean_dependency():
    return get_target_source_from_pickle("input/new_drug_mean_dependency.pickle")


def load_new_drug_from_exp_dep():
    dep_target, dep_source = load_new_drug_mean_dependency()
    exp_target, exp_source = load_new_drug_mean_expression()
    # print(exp_source.shape)
    concatenated_source = dep_source.join(exp_source, lsuffix='_dep', rsuffix='_exp')
    target = dep_target.loc[concatenated_source.index]

    return target, concatenated_source



# t_pickle_path = r"C:\Users\Nitay\PycharmProjects\cca\cancer_data_dependency_t_cca.pickle"
t_pickle_path = r"cancer_data_dependency_filtered.pickle"


def load_T_cancer_data():
    if os.path.isfile(t_pickle_path):
        pickle_load = pickle.load(open(t_pickle_path, 'rb'))
        return pickle_load[1], pickle_load[0]

    # gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_effect.csv")
    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    expressions_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_expression_full.csv")
    cancer_gene_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\cancer_gene_census.csv")
    cancer_genes = set(cancer_gene_df['Gene Symbol'])

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0")
    expressions_df = expressions_df.set_index("Unnamed: 0")

    gene_effect_df.columns.names = ["gene"]
    expressions_df.columns.names = ["gene"]
    gene_effect_df.index.names = ["cell_line"]
    expressions_df.index.names = ["cell_line"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(columns=clean_gene_name)
    clean_expressions_df = expressions_df.rename(columns=clean_gene_name)

    common_cell_lines = set(clean_gene_effect_df.index).intersection(clean_expressions_df.index)
    # cancer_genes_in_dataset = set(clean_gene_effect_df.columns).intersection(cancer_genes)

    duplicate_filtered_expressions_df = clean_expressions_df.loc[:,
                                        ~clean_expressions_df.columns.duplicated(keep='first')]
    filtered_gene_effect_df = clean_gene_effect_df.filter(items=common_cell_lines, axis=0) \
        .filter(items=set(clean_gene_effect_df.columns).intersection(cancer_genes), axis=1)
    filtered_expressions_df = duplicate_filtered_expressions_df.filter(items=common_cell_lines, axis=0) \
        .filter(items=set(duplicate_filtered_expressions_df.columns).intersection(cancer_genes), axis=1)

    filtered_expressions_df = filtered_expressions_df.loc[:, filtered_expressions_df.std(0) != 0]

    # from sklearn.cross_decomposition import CCA
    # Y = filtered_gene_effect_df.values
    # X = filtered_expressions_df.values
    # cca = CCA(n_components=200)
    # cca.fit(X, Y)
    # X_c = cca.transform(X)
    # filtered_expressions_df = pd.DataFrame(X_c)

    pickle.dump([filtered_expressions_df, filtered_gene_effect_df], open(t_pickle_path, 'wb'))

    return filtered_gene_effect_df, filtered_expressions_df


fusion_pickle_path = r"cancer_data_fusion.pickle"


def load_fusion_mutation_data():
    if os.path.isfile(fusion_pickle_path):
        pickle_load = pickle.load(open(fusion_pickle_path, 'rb'))
        return pickle_load[0], pickle_load[1]

    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    mutations_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_mutations.csv")
    expressions_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_expression_full.csv")

    mutations_df = mutations_df[mutations_df["isDeleterious"].fillna(False)]

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0").T
    expressions_df = expressions_df.set_index("Unnamed: 0").T

    gene_effect_df.columns.names = ["cell_line"]
    expressions_df.columns.names = ["cell_line"]
    gene_effect_df.index.names = ["gene"]
    expressions_df.index.names = ["gene"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(index=clean_gene_name)
    clean_expressions_df = expressions_df.rename(index=clean_gene_name)

    common_genes = set(clean_gene_effect_df.index).intersection(set(mutations_df['Hugo_Symbol']))\
    .intersection(clean_expressions_df.index)

    mutations_cell_line = set(mutations_df['Broad_ID'])

    new_mutations_df = pd.DataFrame(np.zeros((len(common_genes), len(mutations_cell_line))),
                                    columns=mutations_cell_line, index=common_genes)
    for i, row in mutations_df.iterrows():
        cell_line = row["DepMap_ID"]
        gene = row['Hugo_Symbol']
        new_mutations_df.loc[gene, cell_line] = 1

    duplicates = [item for item, count in collections.Counter(list(clean_expressions_df.index)).items() if count > 1]
    fixed_common_genes = common_genes.difference(duplicates)
    duplicate_filtered_expressions_df = clean_expressions_df.loc[~clean_expressions_df.index.duplicated(keep='first')]

    filtered_gene_effect_df = clean_gene_effect_df.filter(items=fixed_common_genes, axis=0)
    filtered_mutations_df = new_mutations_df.filter(items=fixed_common_genes, axis=0)
    filtered_expressions_df = duplicate_filtered_expressions_df \
        .filter(items=fixed_common_genes, axis=0)

    fusion_df = pd.concat([filtered_expressions_df, filtered_mutations_df], axis=1, sort=False)

    pickle.dump([filtered_gene_effect_df, fusion_df], open(fusion_pickle_path, "wb"))

    return filtered_gene_effect_df, fusion_df


mutation_pickle_path = r"cancer_data_mutation_cca10.pickle"


def load_mutation_data():
    if os.path.isfile(mutation_pickle_path):
        pickle_load = pickle.load(open(mutation_pickle_path, 'rb'))
        return pickle_load[0], pickle_load[1]

    gene_effect_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\Achilles_gene_dependency.csv")
    mutations_df = pd.read_csv(r"C:\Users\Nitay\Documents\courses\roded-seminar\CCLE_mutations.csv")

    mutations_df = mutations_df[mutations_df["isDeleterious"].fillna(False)]

    gene_effect_df = gene_effect_df.set_index("Unnamed: 0").T
    gene_effect_df.columns.names = ["cell_line"]
    gene_effect_df.index.names = ["gene"]

    def clean_gene_name(name):
        return name.split("(")[0].strip()

    clean_gene_effect_df = gene_effect_df.rename(index=clean_gene_name)

    common_genes = set(clean_gene_effect_df.index).intersection(set(mutations_df['Hugo_Symbol']))
    mutations_cell_line = set(mutations_df['DepMap_ID'])

    new_mutations_df = pd.DataFrame(np.zeros((len(common_genes), len(mutations_cell_line))),
                                    columns=mutations_cell_line, index=common_genes)
    for i, row in mutations_df.iterrows():
        cell_line = row["DepMap_ID"]
        gene = row['Hugo_Symbol']
        if gene in common_genes and cell_line in mutations_cell_line:
            new_mutations_df.loc[gene, cell_line] = 1

    filtered_gene_effect_df = clean_gene_effect_df.filter(items=common_genes, axis=0)
    filtered_mutations_df = new_mutations_df.loc[new_mutations_df.sum(1) > 0,new_mutations_df.sum(0) > 0]


    from sklearn.cross_decomposition import CCA
    Y = filtered_gene_effect_df.values
    X = filtered_mutations_df.values
    cca = CCA(n_components=10)
    cca.fit(X, Y)
    X_c = cca.transform(X)
    filtered_mutations_df = pd.DataFrame(X_c)

    pickle.dump([filtered_gene_effect_df, filtered_mutations_df], open(mutation_pickle_path, "wb"))

    return filtered_gene_effect_df, filtered_mutations_df


def load_cancer_dependency_with_neighbors():
    pickle_load = pickle.load(open("expr_dependency_with_mean.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_cancer_dependency_groups_with_neighbors():
    pickle_load = pickle.load(open("expr_dependency_with_mean_and_groups.pickle", 'rb'))
    return pickle_load[1], pickle_load[0], pickle_load[3], pickle_load[2]


def load_fake_group():
    pickle_load = pickle.load(open("fake_groups.pickle", 'rb'))
    return pickle_load[1], pickle_load[0], pickle_load[3], pickle_load[2]


def load_cell_line_expression_to_dependency_high_std():
    [dependency_df, expression_df] = pickle.load(open("input/t_expression_to_dependency.pickle", 'rb'))
    high_std_genes = dependency_df.std().nlargest(1000).index
    homogen_exp = expression_df.loc[:, high_std_genes].std() < 0.01
    homogen_exp_genes = set(homogen_exp[homogen_exp].index)
    high_std_genes_clean = high_std_genes.difference(homogen_exp_genes)

    return dependency_df.loc[:, high_std_genes_clean], expression_df.loc[:, high_std_genes_clean]


def get_top_std(target_df, source_df, percent, source_filter=True):
    gene_count = len(target_df.columns)
    top_gene_count = int(percent * gene_count / 100)

    high_std_genes = target_df.std().nlargest(top_gene_count).index
    homogen_exp = source_df.std() < 0.1
    homogen_exp_genes = set(homogen_exp[homogen_exp].index)

    if source_filter:
        expression_genes = high_std_genes.difference(homogen_exp_genes)
    else:
        expression_genes = set(source_df.columns).difference(homogen_exp_genes)

    return target_df.loc[:, high_std_genes], source_df.loc[:, expression_genes]


def load_cell_line_expression_top_std(percent, source_filter=True):
    [dependency_df, expression_df] = pickle.load(open("input/t_expression_to_dependency.pickle", 'rb'))
    return get_top_std(dependency_df, expression_df, percent, source_filter)


def load_cell_line_cancer_data_expression_gene_panel(high_std=True):
    [dependency_df, expression_df] = pickle.load(open("input/t_expression_to_dependency.pickle", 'rb'))
    expression_gene_panel = pickle.load(open("input/expression_gene_panel.pickle", "rb"))
    common_genes = set(expression_gene_panel).intersection(dependency_df.columns)

    if high_std:
        high_std_genes = dependency_df.std().nlargest(1000).index
        common_genes_high = common_genes.intersection(high_std_genes)
        print(len(common_genes_high))

    return dependency_df.loc[:, common_genes], expression_df.loc[:, common_genes]


def load_cell_line_combined_mutation_expression_panels():
    dependency_expression_panel, expression_panel = load_cell_line_cancer_data_expression_gene_panel()
    dependency_mutation_panel, mutation_panel = load_propagated_cancer_mutations_data()

    common_cell_lines = set(dependency_mutation_panel.index).intersection(dependency_expression_panel.index)

    joint_source = expression_panel.loc[common_cell_lines].join(mutation_panel.loc[common_cell_lines], lsuffix="exp_", rsuffix="mut_")

    panel_difference = set(dependency_expression_panel).difference(dependency_mutation_panel)
    dep_expression_panel_diff = dependency_expression_panel.loc[:, panel_difference]
    joint_target = dep_expression_panel_diff.loc[common_cell_lines].join(dependency_mutation_panel.loc[common_cell_lines])

    # panel_union = set(dependency_expression_panel).union(dependency_mutation_panel)
    # assert list(joint_target) == panel_union

    return joint_target, joint_source


def load_cell_line_cancer_data(percentage):
    pickle_load = pickle.load(open(f"t_cancer_data_{percentage}.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_cell_line_cancer_data_gene_panel():
    pickle_load = pickle.load(open(f"input/t_cancer_data_gene_panel_384.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_cell_line_cancer_data_gene_panel_target_only():
    pickle_load = pickle.load(open(f"input/t_cancer_data_gene_panel_384.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_propagated_cancer_mutations_data():
    pickle_load = pickle.load(open(f"input/propagated_374_non_zero_gene_mutations_panel_target.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_propagated_cancer_mutations_data_diff_dependency():
    with open("input/propagated_non_zero_gene_mutations_to_diff_dependency.pickle", "rb") as f:
        [mutations, dependency] = pickle.load(f)

    return dependency, mutations


def load_ic50_data():
    pickle_load = pickle.load(open("input/ic50_374genes_103drugs.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_ic50_data_all_genes():
    pickle_load = pickle.load(open("input/ic50_mutations_from_all_genes_to_drugs.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_mutations_all_genes_to_panel():
    pickle_load = pickle.load(open("input/propagated_non_zero_gene_mutations_gene_panel_16165_384.pickle", 'rb'))
    return pickle_load[1], pickle_load[0]


def load_pred_gene_expression_dependency():
    [expression_df, dependency_df] = pickle.load(open("cancer_data_dependency.pickle", 'rb'))
    mean = dependency_df.mean(axis=1)
    std = dependency_df.std(axis=1)

    label = std.copy()
    label[std > 0.1] = 2
    label[(std < 0.1) & (mean > 0.5)] = 1
    label[(std < 0.1) & (mean < 0.5)] = 0

    assert np.count_nonzero((label != 0) & (label != 1) & (label != 2)) == 0
    label = label.astype("int32")

    return label, expression_df


def load_gene_expression_differentiability_label_q4():
    with open("input/cancer_data_dependency_q4.pickle", 'rb') as f:
        [expression_df, dependency_df] = pickle.load(f)

    # mean = dependency_df.mean(axis=1)
    std = dependency_df.std(axis=1)

    label = std.copy()
    label[std > 0.1] = 1
    label[std < 0.1] = 0

    assert np.count_nonzero((label != 0) & (label != 1)) == 0
    label = label.astype("int32")

    return label, expression_df


def load_gene_expression_zero_one_dependency_q4():
    with open("input/cancer_data_dependency_q4.pickle", 'rb') as f:
        [expression_df, dependency_df] = pickle.load(f)

    mean = dependency_df.mean(axis=1)
    std = dependency_df.std(axis=1)

    mean_lower_std = mean[std < 0.1]
    new_label = mean_lower_std.copy()
    new_label[mean_lower_std < 0.5] = 0
    new_label[mean_lower_std >= 0.5] = 1

    return new_label, expression_df.loc[new_label.index]


# def get_differential_cancer_data():
#     [expression_df, dependency_df] = pickle.load(open("input/cancer_data_dependency.pickle", 'rb'))
#     mean = dependency_df.mean(axis=1)
#     std = dependency_df.std(axis=1)
#
#     differentiable_genes = std > 0.1
#
#     return dependency_df[differentiable_genes], expression_df[differentiable_genes]


def get_cell_line_prediction_differential_genes_q4():
    [dependency_df, expression_df] = \
        pickle.load(open("input/t_expression_to_dependency_diff_genes_q4.pickle", 'rb'))

    return dependency_df, filter_sparse_columns(expression_df, 0.2)


def filter_sparse_columns(df, percent):
    indices = np.where((df != 0).sum(axis=0) > int(df.shape[0] * percent))[0]
    return df.iloc[:, indices]


def get_ccle_expression_dependency_tcga_diff_genes():
    with open("input/t_expression_to_dependency_tcga2diff_genes.pickle", "rb") as f:
        [dependency, expression] = pickle.load(f)

    return dependency, expression


def get_tcga_and_survival_data_with_model():
    with open("input/tcga_expression_and_survival.pickle", "rb") as f:
        [tcga_expression, survival_data] = pickle.load(f)

    with open("output/regression/new_cell_line/nn/tcga_erange(10, 111, 10)_b32_l0.0001_d[1000, 1000, 1000, 1000, 1000, 1000, 1000]_wd_0.0_(1)/model.pickle",
              "rb") as f:
        model = pickle.load(f)

    return tcga_expression, survival_data, model


def get_new_gene_cancer_data_q4():
    with open("input/cancer_data_dependency_q4.pickle", "rb") as f:
        [expression, dependency] = pickle.load(f)

    valid_dep_cell_lines = dependency.T[~dependency.isnull().any()].index
    # (r,c) = expression.shape
    # fake_expression = pd.DataFrame(data=np.random.rand(r, c),
    #                                index=expression.index,columns=expression.columns)
    return dependency.loc[:, valid_dep_cell_lines], expression


def get_new_gene_cancer_data_differentiable_q4():
    with open("input/expression_to_dependency_diff_genes_q4.pickle", "rb") as f:
        [dependency, expression] = pickle.load(f)

    return dependency, expression
    # dependency, expression = get_new_gene_cancer_data_q4()
    # std = dependency.std(axis=1)
    # differential_genes = std[std > 0.1].index
    #
    # return dependency.loc[differential_genes], expression.loc[differential_genes]


def get_tumor_file():
    data = pd.read_csv(r"input/Proteomic data MM_Tumor_UNIPROT and ENTREZ IDs.txt", sep="\t")

    return data.iloc[:, :-4].transpose(copy=True)


if __name__ == "__main__":
    # target_df, source_df = load_fusion_mutation_data()
    # print(target_df.shape)
    # print(source_df.shape)
    # print(list(source_df.columns))
    # load_cell_line_combined_mutation_expression_panels()
    # get_tumor_file()
    load_new_drug_from_exp_dep()