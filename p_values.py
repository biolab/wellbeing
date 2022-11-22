import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Orange.data import Table
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF


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

def get_ranker_scores(data):
    rank_scores = []
    for x in range(100):
        print(x)
        np.random.shuffle(data.Y)
        scores = RReliefF(random_state=0)(data)
        for score in scores:
            rank_scores.append(score)
    rank_scores_clean = [i for i in rank_scores if i != 0.0]
    print(len(rank_scores_clean))
    return rank_scores_clean

def plot_dist(scores):
    plt.figure(figsize=(14,16))
    sns.histplot(data=scores).set(title="reliefni srečkovič")
    plt.show()

def get_p_value(scores):
    vector = np.array(scores)
    threshold = np.percentile(vector, 95)
    print(threshold)
    return threshold

ranker_scores = get_ranker_scores(data)
rez = get_p_value(ranker_scores)
plot_dist(ranker_scores)