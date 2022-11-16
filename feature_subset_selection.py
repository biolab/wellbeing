import numpy as np
from sklearn.metrics import r2_score
from Orange.data import Table, Domain
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF
from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.regression import LassoRegressionLearner
from Orange.preprocess import Preprocess, Normalize
from Orange.evaluation import CrossValidation
from Orange.data.filter import HasClass


"""
CROSS w/ preprocessing 

RF: A008.W = 0.27
    A170.W = 0.62
    SWB.LS = 0.58

LR: A008.W = 0.11 (alpha = 3)
    A170.W = 0.56 (alpha = 0.5)
    SWB.LS = 0.50 (alpha = 0.5)

CROSS w/o preprocessing

RF: A008.W = 0.28
    A170.W = 0.52
    SWB.LS = 0.54
    
LR: A008.W = **negative values** for alpha in range 0:25
    A170.W = **negative values** for alpha in range 0:25
    SWB.LS = 0.33 (alpha = 0) >>> by using higher alpha values we obtain negative result

normalization / cross-validation


    
normalization / preprocessing / cross-validation

SWB.LS
   alpha: 0.07    result: [0.5676902067182332, 0.6135536431883339]
   alpha: 0.1    result: [0.5676902067182332, 0.5758854346201447]
   alpha: 0.13    result: [0.5676902067182332, 0.5321999520909746]
   alpha: 0.16    result: [0.5676902067182332, 0.49780347585300744]
   alpha: 0.19    result: [0.5676902067182332, 0.4651894014137452]
   alpha: 0.22    result: [0.5676902067182332, 0.4341891510694884]

preprocessing / normalization / cross-validation

SWB.LS 
    alpha: 0.09    result: [0.578937205124497, 0.7107627354388759]
    alpha: 0.1    result: [0.578937205124497, 0.706505618131291]
    alpha: 0.15    result: [0.578937205124497, 0.666166045034656]
    alpha: 0.2    result: [0.578937205124497, 0.6047269432248306]

"""

ALPHA = 0.5    # CAPS LOCK KONSTANTE


def normalization(data):
    normalizer = Normalize(norm_type=Normalize.NormalizeBySD)
    normalized_data = normalizer(data)
    # print(normalized_data)
    return normalized_data



def get_top_attributes(method, data):
    scores = method(data)
    score_attr_pairs = []
    for attr, score in zip(data.domain.attributes, scores):
        if np.isnan(score):
            continue
        else:
            t = (score, attr.name)
            score_attr_pairs.append(t)
    score_attr_pairs.sort(key=lambda x: x[0], reverse=True)
    top_factors = score_attr_pairs[:10]
    return top_factors


def rf_top_attributes(data):
    rf_learner = RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)
    scores, variables = rf_learner.score(data)
    ls_scores = []
    for i, j in zip(scores, variables):
        if np.isnan(i):
            continue
        else:
            t = (i, j.name)
            ls_scores.append(t)
    ls_scores.sort(key=lambda x: x[0], reverse=True)
    top_factors = ls_scores[:10]
    return top_factors

def get_all_top_attributes(table):
    relief_top_factors = get_top_attributes(RReliefF(random_state=0), table)
    linear_top_factors = get_top_attributes(UnivariateLinearRegression(), table)
    random_top_factors = rf_top_attributes(table)
    names_relief = [i[1] for i in relief_top_factors]   # extracting names of top factors
    names_linear = [i[1] for i in linear_top_factors]
    names_random = [i[1] for i in random_top_factors]

    #print(relief_top_factors)
    #print(linear_top_factors)
    #print(random_top_factors)

    all_names = set(names_relief+names_linear+names_random)
    return list(all_names)


def accuracy_of_preprocessed_factors(data):
    preprocessor = FeatureSubsetSelection()
    table_only_top_factors = preprocessor(data)
    lasso = LassoRegressionLearner(alpha=ALPHA, fit_intercept=True)
    forest = RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)
    learners = [lasso, forest]

    filter = HasClass()
    clean_table_only_top_factors = filter(table_only_top_factors)

    """
    r2_scores = []
    for learner in learners:
        model = learner[0]
        y_true = clean_table_only_top_factors.Y
        y_pred = model(clean_table_only_top_factors)
        score = r2_score(y_true, y_pred)
        r2_scores.append(score)
    print(r2_scores)
    """

    model1 = lasso(clean_table_only_top_factors)
    y_true1 = clean_table_only_top_factors.Y
    y_pred1 = model1(clean_table_only_top_factors)
    score1 = r2_score(y_true1, y_pred1)

    model2 = forest(clean_table_only_top_factors)
    y_true2 = clean_table_only_top_factors.Y
    y_pred2 = model2(clean_table_only_top_factors)
    score2 = r2_score(y_true2, y_pred2)
    print(score1, score2)
    return score1, score2


class FeatureSubsetSelection(Preprocess):
    def __call__(self, table: Table) -> Table:
        factor_names = get_all_top_attributes(table)
        attrs = table.domain.attributes                    # extracting names from domain
        l_attr = []
        for attr in attrs:
            if attr.name in factor_names:
                l_attr.append(attr)

        #l_attr = [attr for attr in attrs if attr.name in factor_names]

        domain = Domain(l_attr, table.domain.class_vars, table.domain.metas)    # domain only with l_attr
        table_only_top_factors = table.transform(domain)              # return a table which contains only l_attr columns
        return table_only_top_factors

    """preprocessor / normalization / cross-validation"""
        # normi = normalization(table_only_top_factors)
        # return normi


def cross_validation(data):
    # regression = LinearRegressionLearner()
    lasso = [LassoRegressionLearner(alpha=ALPHA, fit_intercept=True)]
    forest = [RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)]
    learners = [forest, lasso]
    learners_scores = []

    for learner in learners:
        filter = HasClass()
        with_class = filter(data)                   # remove values without a class (target variable)
        cross = CrossValidation(k=len(with_class))

        result = cross(with_class, learner, preprocessor=FeatureSubsetSelection())  # preprocessor=FeatureSubsetSelection()
        y_true = result.actual
        y_pred = result.predicted[0]

        r2 = r2_score(y_true, y_pred)
        learners_scores.append(r2)
        # print(f"R2: {round(r2, 3)}")
    print(learners_scores)
    return learners_scores


if __name__ == "__main__":
    data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_SWB.LS_selected.pkl")

    # preprocess_table(data)
    # print(preprocess_table.domain)
    # preprocess = FeatureSubsetSelection()
    # pre_data = preprocess(data)         # putting this in cross validation, we obtain workfow in orange
    # cross = cross_validation(data)
    # acc = accuracy_of_preprocessed_factors(data)

    """normalization / cross-validation"""  # remove preprocessor from cross and normalization from preprocessor
    # normi = normalization(data)
    # cross = cross_validation(normi)

    results = []
    for alpha in range(7, 23, 3):
        ALPHA = alpha/100
        cross = cross_validation(data)
        results.append((ALPHA, cross))

    for alpha, result in results:
        print(f"alpha: {alpha}    result: {result}")
