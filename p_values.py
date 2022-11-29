import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from Orange.data import Table
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF
from Orange.regression.random_forest import RandomForestRegressionLearner

from feature_subset_selection import relief_top_attributes, linear_top_attributes, rf_top_attributes

data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl")
"""
def get_relief_scores(data):
    rank_scores = []
    for x in range(100):
        np.random.shuffle(data.Y)
        method = RReliefF(random_state=0)
        scores = method(data)
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [x for x in rank_scores if x != 0.0]
    print(len(rank_scores_clean))
    return rank_scores_clean

def get_linear_scores(data):
    rank_scores = []
    for x in range(100):
        np.random.shuffle(data.Y)
        method = UnivariateLinearRegression()
        scores = method(data)
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [x for x in rank_scores if not np.isnan(x)]
    print(len(rank_scores_clean))
    return rank_scores_clean
"""


def get_ranker_scores(ranker, data):
    rank_scores = []
    for x in range(100):
        np.random.shuffle(data.Y)
        scores = ranker(data)
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [x for x in rank_scores if not np.isnan(x)]         # x != 0.0

    print(len(rank_scores_clean))
    return rank_scores_clean


def get_forest_scores(data):
    rank_scores = []
    for x in range(100):
        np.random.shuffle(data.Y)
        method = RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)
        scores = method.score(data)[0]
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [x for x in rank_scores if not np.isnan(x)]  # x != 0.0

    print(len(rank_scores_clean))
    return rank_scores_clean

def plot_dist(scores):
    plt.figure(figsize=(14,16))
    sns.histplot(data=scores).set(title="regresijski srečkovič")
    plt.show()


def calculate_p_value(distribution, top_score):
    """
    cnt = count the number of values in the distribution that are bigger than sample
    p_val = cnt / len(distribution)
    """
    cnt = 0
    for score in distribution:
        if score >= top_score:
            cnt += 1
    p_val = cnt / len(distribution)
    return p_val


def get_ranker_p_values(random_scores, top_factors):
    p_values = {}
    for score, att_name in top_factors:
        p_val = calculate_p_value(random_scores, score)
        # p_values.append(p_val)
        p_values[att_name] = p_val
    return p_values


def get_p_values_for_top_factors(data):
    relief_top_factors = relief_top_attributes(data)
    linear_top_factors = linear_top_attributes(data)
    random_top_factors = rf_top_attributes(data)

    all_top_factors = [relief_top_factors, linear_top_factors, random_top_factors]

    ranker_scores = []
    for ranker in [RReliefF(random_state=0), UnivariateLinearRegression()]:
        ranker_scores.append(get_ranker_scores(ranker, data))
    # ranker_scores.append(get_relief_scores(data))
    # ranker_scores.append(get_linear_scores(data))
    ranker_scores.append(get_forest_scores(data))


    """
    rank_method_names = ['R', 'L', 'F']  # ustvarim seznam kratic za metode rangiranja
    seznamcek = []
    # zdruzim skupaj seznam nakljucnih vrednosti, seznam top faktorjev in ime ranker metode
    for random_scores, top_factors, method_name in zip(ranker_scores, all_top_factors, rank_method_names):
        p_values = get_ranker_p_values(random_scores, top_factors)  # dobim slovar [att_name] --> att p-value
        for att_name, p_val in p_values.items():
            seznamcek.append(p_val)
    print(seznamcek)
    return seznamcek
    """


    rank_method_names = ['R', 'L', 'F']             # ustvarim seznam kratic za metode rangiranja
    slovarcek = {}
    # zdruzim skupaj seznam nakljucnih vrednosti, seznam top faktorjev in ime ranker metode
    for random_scores, top_factors, method_name in zip(ranker_scores, all_top_factors, rank_method_names):
        p_values = get_ranker_p_values(random_scores, top_factors)  # dobim slovar [att_name] --> att p-value
        for att_name, p_val in p_values.items():
            slovarcek[att_name] = (p_val, method_name)              # dobim slovar [att_name] --> method_name
    print(slovarcek)
    return slovarcek



def google(ime_datoteke):
    google = True
    if google:
        with open(f'{ime_datoteke}.csv') as f:
            tabela_str = f.read()
        tabela_str = tabela_str.replace(',', ':').replace('.', ',')
        with open(f'{ime_datoteke}.txt', 'w') as f:
            f.write(tabela_str)
        print(f"za google zapisano v {ime_datoteke}.txt")


get_p_values_for_top_factors(data)


"""
rez = get_p_value(ranker_scores)
plot_dist(ranker_scores)
"""