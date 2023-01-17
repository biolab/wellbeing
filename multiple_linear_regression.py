import pandas as pd
import numpy as np

from Orange.data import Table
from Orange.classification import SimpleTreeLearner
from Orange.preprocess import Impute, Model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

FILEMAP = {
    r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl': [
        'ilc.hcmp03', 'isoc.ci.in.h', 'ilc.chmd04', 'SH.PRG.ANEM', 'hlth.silc.01',
        'SP.DYN.TO65.MA.ZS', 'lfso.16elvncom', 'EG.ELC.RNEW.ZS', 'env.ac.cur', 'WJQ.LMI',
        'SE.PRM.ENRL.TC.ZS'
    ],
    r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A170.W_selected.pkl': [
    'ilc.di11d', 'ilc.mddd17', 'isoc.r.iacc.h', 'ilc.mdes01', 'ilc.mddd13', 'SI.DST.10TH.10',
    'WJQ.LHI', 'SAF.H', 'WJQ.LMI'
    ],
    r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_SWB.LS_selected.pkl': [
    'isoc.ci.in.h', 'ilc.mddd13', 'WJQ.E', 'ilc.mdes02', 'ilc.mdho07', 'isoc.ci.cm.h',
    'SH.PRG.ANEM', 'ilc.mded03', 'NY.GNP.PCAP.PP.KD', 'educ.uoe.enra29', 'icw.sr.03'
    ]
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
    y = y.drop(index = nan_index).to_numpy()
    x = x.drop(index = nan_index)

    names = [attr.name for attr in data.domain.attributes]
    x.columns = names
    return x, y

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
    coef = model.coef_[0]

    print(f'coefficient of determination: {round(r_sq, 3)}')
    print(f'intercept: {round(intercept, 3)}')
    print(f'coefficients: {np.round_(coef, 3)}')

    for column_name, coef_val in zip(var_selection.columns, coef):
        print(f"{column_name}: {round(coef_val, 3)}")

for filepath, var_names in FILEMAP.items():
    data, y = create_df(filepath)
    data = standardization(data)
    print(f'\n\nresults for filepath: {filepath}: \n\n')
    multiple_regression(data, y, var_names)