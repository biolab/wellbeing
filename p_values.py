import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from Orange.data import Table
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF
from Orange.regression.random_forest import RandomForestRegressionLearner


"""
i. extract class_var from orange table
    class_values = data.Y

ii. shuffle the values of class_var inside the list
     np.random.shuffle(data.Y)

iii. on shuffled class_var calculate rrelief scores and  save new ranking scores in list
    rank_scores = []
    scores = RReliefF(data)
    for score in scores:
        rank_scores.append(score)

iv. draw the distribution of rrelieff scores using histogram and get threshold

"""

data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl")


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

def plot_dist(scores):
    plt.figure(figsize=(14,16))
    sns.histplot(data=scores).set(title="regresijski srečkovič")
    plt.show()

def get_p_value(scores, percentile):
    vector = np.array(scores)
    threshold = np.percentile(vector, percentile)
    print(threshold)
    return threshold

def racunalnik(data):
    ranker_scores = []
    for ranker in [RReliefF(), UnivariateLinearRegression()]:
        ranker_scores.append(get_ranker_scores(ranker, data))
    ranker_scores.append(get_forest_scores(data))

    result = []
    for scores in ranker_scores:
        p_vals = []
        for percentile in [95, 99, 99.9]:
            p_vals.append(get_p_value(scores, percentile))
        result.append(p_vals)
    print(result)
    return result

def sparovcek(list_listov):
    df = pd.DataFrame(columns=['ranker', 'p-value'])
    rank_method_names = ['relief', 'univariate', 'forest']
    for name, list in zip(rank_method_names, list_listov):
        for value in list:
            dict = {'ranker': name, 'p-value': value}
            df = df.append(dict, ignore_index=True)

    df.to_csv('p-val_A008W.csv', index=False)


pozeni = racunalnik(data)
konec = sparovcek(pozeni)

"""
rez = get_p_value(ranker_scores)
plot_dist(ranker_scores)
"""