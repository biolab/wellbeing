import numpy as np
from sklearn.metrics import r2_score
from Orange.data import Table, Domain, DiscreteVariable
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF
from Orange.regression import RandomForestRegressionLearner
from Orange.preprocess import Preprocess
from Orange.evaluation import CrossValidation
from Orange.data.filter import HasClass


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
    rf_learner = RandomForestRegressionLearner(random_state=0)
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

class FeatureSubsetSelection(Preprocess):

    def __call__(self, table: Table) -> Table:
        relief_top_factors = get_top_attributes(RReliefF(), table)
        linear_top_factors = get_top_attributes(UnivariateLinearRegression(), table)
        random_top_factors = rf_top_attributes(table)
        names_relief = [i[1] for i in relief_top_factors]  # extracting names of top factors
        names_linear = [i[1] for i in linear_top_factors]
        names_random = [i[1] for i in random_top_factors]
        attrs = table.domain.attributes                    # extracting names from domain
        l_attr = []
        for attr in attrs:
            if attr.name in names_relief:
                l_attr.append(attr)
            elif attr.name in names_linear:
                l_attr.append(attr)
            elif attr.name in names_random:
                l_attr.append(attr)

        #l_attr = [attr for attr in attrs if attr.name in names_relief+names_linear+names_random]


        domain = Domain(l_attr, table.domain.class_vars, table.domain.metas)
        return table.transform(domain)

def cross_validation(data):
    """Accepts orange Table"""
    forest = RandomForestRegressionLearner()
    learners = [forest]
    cross = CrossValidation(k=10)

    filter = HasClass()
    with_class = filter(data)           # remove values without a class (target variable)

    result = cross(with_class, learners, preprocessor=FeatureSubsetSelection())

    y_true = result.actual
    y_pred = result.predicted[0]

    r2 = r2_score(y_true, y_pred)
    print(f"R2: {round(r2, 3)}")
    return r2


if __name__ == "__main__":
    data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_target_selected.pkl")
    # preprocess_table(data)
    # print(preprocess_table.domain)
    preprocessor = FeatureSubsetSelection()  # narediš instanco
    preprocessed_data = preprocessor(data)
    cross = cross_validation(data)
    print(len(preprocessed_data.domain.attributes))
