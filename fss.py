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

COUNTRIES RANKINGS:

FSSN: alpha: 0.02    result: [0.7813298763149761, 0.7645338050294848]
NORM: alpha: 0.005    result: [0.794373026251077, 0.8781621888492757]
   
    
FSS: cross-validation w/ feature selection

    A008.W: alpha: 0.06    result: [0.2745338786986795, 0.37402929446981015]
    A170.W: alpha: 0.4    result: [0.6234367513732202, 0.5709059009289654]
    SWB.LS: alpha: 0.5    result: [0.578937205124497, 0.4982400941730326]


NORM: normalize data - cross-validation

    A008.W: alpha: 0.04    result: [0.27414200658144317, 0.26117096682090146]
    A170.W: alpha: 0.06    result: [0.5225549875749862, 0.5229903312967453]
    SWB.LS: alpha: 0.04    result: [0.535172601752893, 0.6387198800953162]
    ranking: alpha: 1.2    result: [0.6025372113236223, 0.56280250438162]

NFSS: normalize data - cv w/ feature selection

    A008.W: alpha: 0.02    result: [0.2745338786986795, 0.3466984099730276]
    A170.W: alpha: 0.04    result: [0.6229451343114472, 0.4914933272401233]
    SWB.LS: alpha: 0.06    result: [0.5676902067182332, 0.625220461605805]
    
FSSN: cv with normalization and feature selection

    A008.W: alpha: 0.04    result: [0.2745338786986795, 0.32050079072966553]
    A170.W: alpha: 0.06    result: [0.6231218613727252, 0.4937621684794109]
    SWB.LS: alpha: 0.1     result: [0.5682499001732606, 0.7065056358036739]
    
"""

# this is used in accuracy calculation and cross_validation
# we change it below to run the code for different values of alpha.
ALPHA = 0.5                 # CAPS LOCK KONSTANTE

def normalization(data):
    """ normalize the data. """
    normalizer = Normalize(norm_type=Normalize.NormalizeBySD)
    normalized_data = normalizer(data)
    return normalized_data


# ATTENTION:
#   IN ORDER TO GET EXACTLY 30 FACTORS, NECESSARY FOR CREATING DATA TABLE,
#   CHANGE THE RANGE OF SCORES IN EACH RANKING METHOD SEPERATELY AS WRITTEN IN COMMENTS BESIDES FUNCTION


def relief_top_attributes(data):    # A008W: 12     A170.W: 11      SWB.LS: 15       rank: 10
    """return 10 top attributes according to RreliefF."""
    scores = RReliefF(random_state=0)(data)
    ls_scores = []
    for attr, score in zip(data.domain.attributes, scores):
        if np.isnan(score):
            continue
        else:
            t = (score, attr.name)
            ls_scores.append(t)
    ls_scores.sort(key=lambda x: x[0], reverse=True)
    top_factors = ls_scores[:10]        # change this according to the target
    return top_factors

def linear_top_attributes(data):    # A008W: 10     A170.W: 10      SWB.LS: 10      rank: 10
    """return 10 top attributes according to UnivariateLinearRegression."""
    scores = UnivariateLinearRegression()(data)
    ls_scores = []
    for attr, score in zip(data.domain.attributes, scores):
        if np.isnan(score):
            continue
        else:
            t = (score, attr.name)
            ls_scores.append(t)
    ls_scores.sort(key=lambda x: x[0], reverse=True)
    top_factors = ls_scores[:10]        # change this according to the target
    return top_factors


def rf_top_attributes(data):        # A008W: 11     A170.W: 10      SWB.LS: 10       RANK: 10
    """return 10 top attributes according to RandomForestRegressionLearner."""
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
    top_factors = ls_scores[:10]        # change this according to the target
    return top_factors

def get_all_top_attributes(table):              # A008.W: 29    A170.W: 28  SWB.LS: 29  RANK: 30
    """
    for each of the 3 methods get top 10 attributes.
    return the list of top attributes, from which duplicates are removed.
    """
    relief_top_factors = relief_top_attributes(table)
    linear_top_factors = linear_top_attributes(table)
    random_top_factors = rf_top_attributes(table)

    print(f'relief: {[name for (val, name) in relief_top_factors]}')
    print(f'linear: {[name for (val, name) in linear_top_factors]}')
    print(f'random forest: {[name for (val, name) in random_top_factors]}')


    print(relief_top_factors, linear_top_factors, random_top_factors)

    # extracting names of top factors
    all_names = set()
    for factor_list in [relief_top_factors, linear_top_factors, random_top_factors]: # For each factor list
        for factor in factor_list:      # for each factor in the list
            all_names.add(factor[1])    # add the name to the set

    return list(all_names)


def accuracy_of_preprocessed_factors(data):
    """
    get top factors, calculate the accuracy of two learners: LassoRegression and RandomForestRegression.
    """
    # get table only containing top attributes
    preprocessor = FeatureSubsetSelection()
    table_only_top_factors = preprocessor(data)

    # remove values without a class (target variable)
    filter = HasClass()
    clean_table_only_top_factors = filter(table_only_top_factors)

    # create two learners
    lasso = LassoRegressionLearner(alpha=ALPHA, fit_intercept=True)
    forest = RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)
    learners = [lasso, forest]

    y_true = clean_table_only_top_factors.Y

    # for each of the two learners, create a model, get a prediction,
    # evaluate the prediction to get an accuracy score, return both scores
    scores = []
    for learner in learners:
        model = learner(clean_table_only_top_factors)
        y_pred = model(clean_table_only_top_factors)
        score = r2_score(y_true, y_pred)
        scores.append(score)

    return scores


class FeatureSubsetSelection(Preprocess):
    def __call__(self, table: Table) -> Table:
        """
        input original Orange Table
        calculate top attributes (factors)
        return a smaller Orange Table which contains only the top attributes (columns)
        """
        top_factor_names = get_all_top_attributes(table)
        # get a list of attributes from the Orange Table whose name is in the top_factor_names.
        attrs = table.domain.attributes
        l_attr = [attr for attr in attrs if attr.name in top_factor_names]

        domain = Domain(l_attr, table.domain.class_vars, table.domain.metas)    # domain only with l_attr
        table_only_top_factors = table.transform(domain)                        # return a table which contains only l_attr columns
        return table_only_top_factors


def cross_validation(data, preprocessor):
    """
    preprocessor: string for which type of preprocessing to run (ALL / FSS / FSSN)
    for the dataset, run cross validation for two types of learners,
    return their R2 scores.
    """

    # define what type of preprocessing will be used in cross-validation.
    if preprocessor == 'ALL':
        preprocessor = None
    elif preprocessor == "FSS":
        preprocessor = FeatureSubsetSelection()
    elif preprocessor == 'FSSN':
        preprocessor_types = [Normalize(norm_type=Normalize.NormalizeBySD), FeatureSubsetSelection()]
        preprocessor = PreprocessorList(preprocessor_types)  # cross-validation with both preprocessors
    else:
        raise ValueError('ni taprav preprocessor')

    # regression = LinearRegressionLearner()
    lasso = [LassoRegressionLearner(alpha=ALPHA, fit_intercept=True)]
    forest = [RandomForestRegressionLearner(n_estimators=100, min_samples_split=5, random_state=0)]
    learners = [forest, lasso]
    learners_scores = []

    # remove values without a class (target variable)
    filter = HasClass()
    with_class = filter(data)

    # define cross-validation parameters
    cross = CrossValidation(k=len(with_class))

    for learner in learners:
        # run cross validation on the data for these processors
        result = cross(with_class, learner, preprocessor=preprocessor)  # preprocessor=FeatureSubsetSelection()
        y_true = result.actual
        y_pred = result.predicted[0]

        # calculate r2 score
        r2 = r2_score(y_true, y_pred)
        learners_scores.append(r2)
        # print(f"R2: {round(r2, 3)}")

    return learners_scores


def model_accuracy(data, type):
    # cross validation with no normalisation nor fss
    if type == "ALL":
        return cross_validation(data, "ALL")
    # cross-validation which performs fss
    elif type == "FSS":
        return cross_validation(data, "FSS")
    # cross-validation within which normalisation and fss is performed
    elif type == "FSSN":
        return cross_validation(data, 'FSSN')
    # normalise data first, then run cross-validation with fss.
    elif type == "NFSS":
        normi = normalization(data)
        return cross_validation(normi, "FSS")
    # normalize data first, then run cross-validation which performs no normalisation nor FSS
    elif type == "NORM":
        normi = normalization(data)
        return cross_validation(normi, 'ALL')
    # fss first, then normalise, then run cross-validation
    elif type == "XNFSS":
        preprocessed_data = FeatureSubsetSelection()(data)
        normi = normalization(preprocessed_data)
        return cross_validation(normi, "ALL")
    else:
        raise ValueError(f'Wrong type: {type}')


if __name__ == "__main__":
    data = Table("C:\\Users\\irisc\\Documents\\FRI\\blaginja\\FRI-blaginja\\input data\\ranking_survey.pkl")

    type = "FSSN"           # change this to run in a different mode.

    results = []
    for alpha in [0.1, 0.5]:
        ALPHA = alpha
        result = model_accuracy(data, type)
        results.append((alpha, result))

    for alpha, result in results:
        print(f"alpha: {alpha}    result: {result}")

