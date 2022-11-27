import numpy as np
import pandas as pd
import json
from sklearn.metrics import r2_score
from Orange.data import Table, Domain
from Orange.preprocess.score import UnivariateLinearRegression, RReliefF
from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.regression import LassoRegressionLearner
from Orange.preprocess import Preprocess, Normalize, PreprocessorList
from Orange.evaluation import CrossValidation
from Orange.data.filter import HasClass


"""
ALL: cross-validation

A008.W 
    alpha: 0.5    result: [0.2746688840615755, -266.01053568610115] 
    alpha: 1      result: [0.2746688840615755, -30.60461446063249]
    alpha: 1.5    result: [0.2746688840615755, -3.057001471266874]

A170.W
    alpha: 0.1    result: [0.5151058286358667, -752.6556680441837]
    alpha: 0.5    result: [0.5151058286358667, -2539.8145648342825]
    alpha: 1    result: [0.5151058286358667, -2176.649175605183]
    
SWB.LS    
    **alpha: 0    result: [0.542922061779767, 0.33223494884996607]
    alpha: 0.02    result: [0.542922061779767, -1.5106110931623928]
    alpha: 0.04    result: [0.542922061779767, -4.160150398263308]
    
    
FSS: cross-validation w/ feature selection

A008.W
    alpha: 0.05    result: [0.2745338786986795, 0.3447987850441635]
    **alpha: 0.06    result: [0.2745338786986795, 0.37402929446981015]**
    alpha: 0.1    result: [0.2745338786986795, 0.32009137208384486]

A170.W
    alpha: 0.2    result: [0.6234367513732202, 0.5588054630860201]
    **alpha: 0.4    result: [0.6234367513732202, 0.5709059009289654]**
    alpha: 0.6    result: [0.6234367513732202, 0.5315705374565577] 

SWB.LS
    alpha: 0.2    result: [0.578937205124497, 0.48960884351948386]
    alpha: 0.3    result: [0.578937205124497, 0.49023461206717256]
    **alpha: 0.5    result: [0.578937205124497, 0.4982400941730326]**


NORM: normalize data - cross-validation

A008.W
    alpha: 0.02    result: [0.27414200658144317, 0.1719769046759434]
    **alpha: 0.04    result: [0.27414200658144317, 0.26117096682090146]**
    alpha: 0.06    result: [0.27414200658144317, 0.23307052396973404]

A170.W 
    alpha: 0.04    result: [0.5225549875749862, 0.44957412998575186]
    **alpha: 0.06    result: [0.5225549875749862, 0.5229903312967453]**
    alpha: 0.08    result: [0.5225549875749862, 0.4810275889211615]   

SWB.LS
    alpha: 0.02    result: [0.535172601752893, 0.6262890012130881]
    **alpha: 0.04    result: [0.535172601752893, 0.6387198800953162]**
    alpha: 0.06    result: [0.535172601752893, 0.6001022875617558]

ranking
    alpha: 1    result: [0.6025372113236223, 0.5551816524250228]
    **alpha: 1.2    result: [0.6025372113236223, 0.56280250438162]**
    alpha: 1.4    result: [0.6025372113236223, 0.5594251413200317]

NFSS: normalize data - cv w/ feature selection

A008.W
    alpha: 0.01    result: [0.2745338786986795, 0.28314783814886135]
    **alpha: 0.02    result: [0.2745338786986795, 0.3466984099730276]**
    alpha: 0.04    result: [0.2745338786986795, 0.324476460317514]
    alpha: 0.06    result: [0.2745338786986795, 0.23769961233662762]

A170.W
    alpha: 0.02    result: [0.6229451343114472, 0.4713305550131984]
    **alpha: 0.04    result: [0.6229451343114472, 0.4914933272401233]**
    alpha: 0.06    result: [0.6229451343114472, 0.4635202691562629]

SWB.LS
    alpha: 0.02    result: [0.5676902067182332, 0.606967907171617]
    alpha: 0.04    result: [0.5676902067182332, 0.6251625134692995]
    **alpha: 0.06    result: [0.5676902067182332, 0.625220461605805]**
    alpha: 0.08    result: [0.5676902067182332, 0.6002252042919891]

XNFSS: feature selection - normalize data - cv >>> PREVERI!!

A008.W
    alpha: 0.02    result: [0.2745338786986795, 0.3190194616453528]
    **alpha: 0.04    result: [0.2745338786986795, 0.32050079072966553]**
    alpha: 0.06    result: [0.2745338786986795, 0.2376603344126128]
    
A170.W
    alpha: 0.03    result: [0.6231218613727252, 0.47735573831608025]
    **alpha: 0.05    result: [0.6231218613727252, 0.49567509095237927]**
    alpha: 0.1    result: [0.6231218613727252, 0.44550222867618605]  
    
SWB.LS 
    alpha: 0.08    result: [0.578937205124497, 0.6987946053062986]
    **alpha: 0.09    result: [0.578937205124497, 0.7107627354388759]**
    alpha: 0.1    result: [0.578937205124497, 0.706505618131291]
    alpha: 0.11    result: [0.578937205124497, 0.6992242145519687]
    
FSSN: cv with normalization and feature selection

A008.W
    alpha: 0.02    result: [0.2745338786986795, 0.3190194616453528]
    **alpha: 0.04    result: [0.2745338786986795, 0.32050079072966553]**
    alpha: 0.06    result: [0.2745338786986795, 0.2376603344126128]

A170.W
    alpha: 0.02    result: [0.6231218613727252, 0.44504615549634796]
    alpha: 0.04    result: [0.6231218613727252, 0.49060008501903407]
    **alpha: 0.06    result: [0.6231218613727252, 0.4937621684794109]**
    alpha: 0.08    result: [0.6231218613727252, 0.475241079998746]

SWB.LS
    alpha: 0.08    result: [0.5682499001732606, 0.6987946306665085]
    **alpha: 0.1     result: [0.5682499001732606, 0.7065056358036739]**
    alpha: 0.12    result: [0.5682499001732606, 0.690980252169229]
    
ranking
    alpha: 2.0    result: [0.6069545852300877, 0.6683099167781903]
    **alpha: 2.2    result: [0.6069545852300877, 0.6694888113113995]**
    alpha: 2.5    result: [0.6069545852300877, 0.6611874967684047]
    alpha: 2.8    result: [0.6069545852300877, 0.655769069241162]

"""

ALPHA = 0.5                 # CAPS LOCK KONSTANTE


def normalization(data):
    normalizer = Normalize(norm_type=Normalize.NormalizeBySD)
    normalized_data = normalizer(data)
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
    top_factors = score_attr_pairs[:11]
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
    top_factors = ls_scores[:11]
    return top_factors

def get_all_top_attributes(table):
    relief_top_factors = get_top_attributes(RReliefF(random_state=0), table)
    linear_top_factors = get_top_attributes(UnivariateLinearRegression(), table)
    random_top_factors = rf_top_attributes(table)

    # shranilnik(relief_top_factors, linear_top_factors, random_top_factors)

    names_relief = [i[1] for i in relief_top_factors]   # extracting names of top factors
    names_linear = [i[1] for i in linear_top_factors]
    names_random = [i[1] for i in random_top_factors]

    print(relief_top_factors, linear_top_factors, random_top_factors)

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
        table_only_top_factors = table.transform(domain)                        # return a table which contains only l_attr columns
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

        preprocessor_types = [Normalize(norm_type=Normalize.NormalizeBySD), FeatureSubsetSelection()]
        list_of_preprocessors = PreprocessorList(preprocessor_types)             # cross-validation with both preprocessors
        result = cross(with_class, learner, preprocessor=list_of_preprocessors)  # preprocessor=FeatureSubsetSelection()
        y_true = result.actual
        y_pred = result.predicted[0]

        r2 = r2_score(y_true, y_pred)
        learners_scores.append(r2)
        # print(f"R2: {round(r2, 3)}")
    print(learners_scores)
    return learners_scores

def shranilnik(list1, list2, list3):
    df = pd.DataFrame(columns=['ranker', 'att', 'score'])
    rank_method_names = ['relief', 'univariate', 'forest']
    for name, list in zip(rank_method_names, [list1, list2, list3]):
        for score, att in list:
            dict = {'ranker': name, 'att': att, 'score': score}
            df = df.append(dict, ignore_index=True)

    df.to_csv('hren.csv', index=False)

    # df = pd.read_csv('hren.csv')
    
    """
    with open('shren.json', 'w') as fp:
        json.dump(dict, fp)       

    
    with open('shren.json', 'r') as fp:
        dict = json.load(fp)    
    """


if __name__ == "__main__":
    data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl")

    # preprocess_table(data)
    # print(preprocess_table.domain)
    preprocess = FeatureSubsetSelection()   # only FFS
    pre_data = preprocess(data)             # putting this in cross validation, we obtain workflow in orange
    # print(pre_data)
    # acc = accuracy_of_preprocessed_factors(data)  # R2 on FSS only

    """normalization / cross-validation"""  # remove preprocessor from cross and normalization from preprocessor
    # normi = normalization(data)
    # cross = cross_validation(normi)

    results = []
    for alpha in [1]:
        ALPHA = alpha
        # cross = cross_validation(data)
        # results.append((ALPHA, cross))

    for alpha, result in results:
        print(f"alpha: {alpha}    result: {result}")
