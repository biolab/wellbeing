import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Orange.data import Table
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF
from Orange.regression.random_forest import RandomForestRegressionLearner

from fss import relief_top_attributes, linear_top_attributes, rf_top_attributes



def get_ranker_scores(ranker, data):
    """
    shuffle Y 100x times and calculate 'random' ranking scores on the same data.
    we will then check how good our real calculated scores are, compared to these 'random' scores.
    """
    rank_scores = []
    for x in range(100):
        np.random.shuffle(data.Y)
        scores = ranker(data)               # obtain ranker scores for randomized target variable column
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [score for score in rank_scores if not np.isnan(score)]     # x != 0.0

    print(len(rank_scores_clean))
    return rank_scores_clean


def get_forest_scores(data):
    """
    shuffle Y 100x times and calculate 'random' ranking scores on the same data.
    we will then check how good our real calculated scores are, compared to these 'random' scores.
    """
    rank_scores = []
    for x in range(100):
        np.random.shuffle(data.Y)
        method = RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)
        scores = method.score(data)[0]
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [score for score in rank_scores if not np.isnan(score)]  # x != 0.0

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
    """
    creating a dictionary, in which key are names of factors and values are their p-values.
    """
    p_values = {}
    for score, att_name in top_factors:
        p_val = calculate_p_value(random_scores, score)
        p_values[att_name] = p_val
    return p_values


def get_p_values_for_top_factors(data):
    # for each ranking method get top attributes.
    relief_top_factors = relief_top_attributes(data)
    linear_top_factors = linear_top_attributes(data)
    random_top_factors = rf_top_attributes(data)

    all_top_factors = [relief_top_factors, linear_top_factors, random_top_factors]


    # for each ranking method, generate 'random' ranking scores.
    ranker_scores = []
    for ranker in [RReliefF(random_state=0), UnivariateLinearRegression()]:
        ranker_scores.append(get_ranker_scores(ranker, data))
    ranker_scores.append(get_forest_scores(data))

    rank_method_names = ['R', 'L', 'F']             # create a list of initials for ranking methods.

    slovarek = {}                                   # hold final p value for each attribute name.
    # combine together a list of random values, list of top factors and the name of ranking method
    for random_scores, top_factors, method_name in zip(ranker_scores, all_top_factors, rank_method_names):
        p_values = get_ranker_p_values(random_scores, top_factors)  # dobim slovar [att_name] --> att p-value
        
        # we iterate through every attribute name and its corresponding p-value.
        # for each one we check:
        #   if 'slovarek' does not yet hold a value this attribute name, we add it.
        #   if 'slovarek' already holds a value for this attribute name, we overwrite it if the current p value is better then the existing one.
        for att_name, p_val in p_values.items():
            if att_name not in slovarek.keys():
                slovarek[att_name] = (p_val, method_name)  # dobim slovar [att_name] --> (p value, method_name)
            else:
                obstojeci_p_val, obstojeca_metoda = slovarek[att_name]
                if p_val < obstojeci_p_val:             
                    print(f'Pri {att_name} prepisujem {obstojeci_p_val} od {obstojeca_metoda} z {p_val} od {method_name}')
                    slovarek[att_name] = (p_val, method_name)  # dobim slovar [att_name] --> (p value, method_name)

    print(slovarek)
    return slovarek



def google(ime_datoteke):
    """we're not currently using this anywhere. """
    google = True
    if google:
        with open(f'{ime_datoteke}.csv') as f:
            tabela_str = f.read()
        tabela_str = tabela_str.replace(',', ':').replace('.', ',')
        with open(f'{ime_datoteke}.txt', 'w') as f:
            f.write(tabela_str)
        print(f"za google zapisano v {ime_datoteke}.txt")

if __name__ == "__main__":
    data = Table("C:\\Users\\irisc\\Documents\\FRI\\blaginja\\FRI-blaginja\\input data\\ranking_survey.pkl")
    slovarcek_p_values = get_p_values_for_top_factors(data)


"""
rez = get_p_value(ranker_scores)
plot_dist(ranker_scores)
"""