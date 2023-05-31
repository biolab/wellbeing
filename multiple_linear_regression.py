import sys
import pandas as pd

from Orange.data import Table
from Orange.classification import SimpleTreeLearner
from Orange.preprocess import Impute, Model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

FILEMAP = {
    r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\input data\A008.W.pkl': [
        'ilc.chmd04', 'SH.PRG.ANEM', 'ilc.mded03', 'hlth.silc.01',
        'SP.DYN.TO65.MA.ZS', 'WJQ.LHI', 'EG.ELC.RNEW.ZS', 'SCO.SS', 'WJQ.ER'
    ],
    r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\input data\A170.W.pkl': [
    'ilc.mddd11', 'ilc.mdes01', 'SI.DST.10TH.10', 'WJQ.LHI', 'SAF.H',
    'SAF.FSA', 'SCO.SS', 'WJQ.LMI', 'WJQ.ER'
    ],
    r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\input data\SWB.LS.pkl': [
    'ilc.mddd13', 'SCO.SS', 'ilc.mdes02', 'ilc.mdho07', 'SH.PRG.ANEM', 'ilc.mded03',
    'NY.GNP.PCAP.PP.KD', 'hlth.silc.01', 'educ.uoe.enra29'
    ],
    r'C:\Users\irisc\Documents\FRI\blaginja\FRI-blaginja\input data\ranking_survey_intersection.pkl': [
    'SP.DYN.LE00.IN', 'ilc.lvho05a', 'WJQ.LHI', 'HH.HA', 'WJQ.ER', 'WJQ.E'
    ],
    r'C:\Users\irisc\Documents\FRI\blaginja\FRI-blaginja\input data\ranking_survey_difference.pkl': [
    'ilc.lvho06', 'SH.DYN.NCOM.ZS', 'lfso.16elvncom', 'SCO.SS', 'SAF.RD', 'SAF.H', 'NY.ADJ.DCO2.GN.ZS',
    'lfso.04avpoisco', 'lfso.04avpona11', 'SP.DYN.AMRT.MA', 'SP.DYN.AMRT.FE'
    ],
    r'C:\Users\irisc\Documents\FRI\blaginja\FRI-blaginja\input data\ranking_survey.pkl': [
    'SP.DYN.LE00.IN', 'SH.DYN.NCOM.ZS', 'ilc.lvho05a', 'WJQ.LHI',
    'SCO.SS', 'HH.HA', 'WJQ.ER', 'WJQ.E'
    ],
}

def impute_missing_values(data):
    imputer = Impute(method=Model(SimpleTreeLearner()))
    impute_data = imputer(data)
    return impute_data

def create_df(filepath):
    data = Table(filepath)
    imputed_data = impute_missing_values(data)

    x = pd.DataFrame(imputed_data.X)
    y = pd.DataFrame(data.Y)

    # remove rows where y is NaN
    nan_index = pd.isnull(y).any(1).to_numpy().nonzero()[0]
    y = y.drop(index = nan_index).to_numpy()        # convert to np array, as standardization function requires it
    x = x.drop(index = nan_index)

    names = [attr.name for attr in data.domain.attributes]  # obtain the list of attribute names
    x.columns = names                                       # set the names of columns as attributes names
    return x, y


def multicollinearity_test(data, var_names):
    df = data[var_names]
    vif = pd.DataFrame()
    vif['variables'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif)
    return vif

def standardization(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
    print(standardized_df)
    return standardized_df

def multiple_regression(df, y, var_names):
    var_selection = df[var_names]
    x = var_selection.to_numpy()

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    intercept = model.intercept_[0]
    coef = model.coef_[0].tolist()
    coef = [round(co, 3) for co in coef]

    print(f'coefficient of determination: {round(r_sq, 3)}')
    print(f'intercept: {round(intercept, 3)}')
    print(f'coefficients: {coef}')

    for column_name, coef_val in zip(var_selection.columns, coef):
        print(f"{column_name}: {coef_val}")

    return coef


def create_csv(var_names, coefs, filepath):
    df = pd.DataFrame()
    df = df.assign(ID = var_names)
    df2 = pd.read_csv(r'C:\Users\irisc\Documents\FRI\blaginja\FRI-blaginja\input data\indicators_shorter.csv')
    df2 = df2.set_index('index')
    descriptions = [df2.loc[var_name, 'description'] for var_name in var_names]
    df = df.assign(DESCRIPT = descriptions)
    df = df.assign(COEF = coefs)
    potica = filepath.split('\\')[-1]
    potica = f'export data\{potica}_regression_coeffs.csv'
    df.to_csv(potica)

for filepath, var_names in FILEMAP.items():
    data, y = create_df(filepath)
    data = standardization(data)
    multicollinearity_test(data, var_names)
    print(f'\n\nresults for filepath: {filepath}: \n\n')
    coefs = multiple_regression(data, y, var_names)
    create_csv(var_names, coefs, filepath)

