import pandas as pd
import numpy as np

from Orange.data import Table
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def create_df():
    filepath = r'C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl'
    data = Table(filepath)
    x_cleaned = np.where(np.isnan(data.X), 0, data.X)
    x = pd.DataFrame(x_cleaned)

    y = np.where(np.isnan(data.Y), 0, data.Y)


    names = [attr.name for attr in data.domain.attributes]
    x.columns = names
    df = x
    return df, y


def standardization(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
    print(standardized_df)
    return standardized_df

def multiple_regression(df, y):
    var_selection = df[['ilc.hcmp03', 'isoc.ci.in.h', 'ilc.chmd04', 'SH.PRG.ANEM', 'hlth.silc.01',
                        'SP.DYN.TO65.MA.ZS', 'lfso.16elvncom', 'EG.ELC.RNEW.ZS', 'env.ac.cur', 'WJQ.LMI',
                        'SE.PRM.ENRL.TC.ZS']]
    x = var_selection.to_numpy()

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    intercept = model.intercept_
    coef = model.coef_

    print(f'coefficient of determination: {r_sq}')
    print(f'intercept: {intercept}')
    print(f'coefficients: {coef}')

    for column_name, coef_val in zip(var_selection.columns, coef):
        print(f"{column_name}: {coef_val}")


data, y = create_df()
data = standardization(data)
multiple_regression(data, y)