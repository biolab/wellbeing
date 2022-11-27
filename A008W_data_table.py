import pandas as pd
from Orange.data import Table
from feature_subset_selection import get_all_top_attributes
from p_values import get_p_values_for_top_factors


def create_df():
    data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl")
    d = pd.DataFrame(data.X)
    dm = pd.DataFrame(data.metas)

    df = pd.concat([dm, d], axis=1)                         # zdruzim x in mete v eno tabelo

    meta_names = data.metas_df.columns.to_list()            # dobim list imen vseh meta stolpcev
    names = [attr.name for attr in data.domain.attributes]  # dobim list imen vseh atributov
    column_names = meta_names + names                       # zdruzim oba lista imen
    df.columns = column_names                               # nastavim imena stolpcev
    d.columns = names                                       # nastavim imena stoplcev na originalnih podatkih
    df = df.set_index('index')          # vrstico dobim z indeksom drzave namesto s stevilko vrstice (npr df.loc['DEU'])

    top_factors = get_all_top_attributes(data)              # dobim list vseh top faktorjev
    df = df[top_factors]                                    # vzamem samo tiste stolpce, ki se nanasajo na top faktorje
    df = df.loc[['SVN', 'AUT', 'DEU']].transpose()          # vzamem samo navedene vrstice in transponiram

    p_values = get_p_values_for_top_factors(data)
    """
    import json
    with open('p_values.json') as f:
        #json.dump(p_values, f)
        p_values = json.load(f)
     """

    df = df.dropna()  # remove NaN values
    att_names = df.index.to_list()                                      # dobim seznam imen vseh atributov
    list_p_values = [p_values[att_name][0] for att_name in att_names]   # dobim seznam vseh p-values
    df.insert(2, 'P-value', list_p_values)                              # p-valuese vstavim v df na mesto 0
    list_method_names = [p_values[att_name][1] for att_name in att_names]
    df.insert(3, 'Scoring method', list_method_names)                   # imena metod vstavim na df na mesto 1
    print(df)

    df2 = pd.read_csv(r'C:\Users\irisc\Documents\FRI\blaginja\indikatorji-krajsi - Sheet1.csv')
    df2 = df2.set_index('index')
    types = [df2.loc[att_name, 'type'] for att_name in att_names]
    df.insert(1, 'Category', types)
    descriptions = [df2.loc[att_name, 'description'] for att_name in att_names]
    df.insert(0, 'Description', descriptions)

    df3 = pd.read_csv(r'C:\Users\irisc\Documents\FRI\blaginja\Kopija od 2111_porocilo_blaginja - A008.W.csv')
    df3 = df3.set_index('ID Indicators')
    units = [df3.loc[att_name, 'unit of measure'] for att_name in att_names]
    df.insert(4, 'Unit of measure', units)

    corr = [df3.loc[att_name, 'higher = better'] for att_name in att_names]
    df.insert(8, 'Higher = Better', corr)
    df = df.astype({'Higher = Better': int})

    # get delta SVN - AUT
    delta_stolpec1 = []
    for att_name in att_names:
        delta = (df.loc[att_name, 'SVN'] - df.loc[att_name, 'AUT']) / max(df.loc[att_name, ['SVN', 'AUT', 'DEU']])
        delta_stolpec1.append(delta)
    df.insert(3, 'Δ AUT', delta_stolpec1)

    # get delta SVN - DEU
    delta_stolpec2 = []
    for att_name in att_names:
        delta = (df.loc[att_name, 'SVN'] - df.loc[att_name, 'DEU']) / max(df.loc[att_name, ['SVN', 'AUT', 'DEU']])
        delta_stolpec2.append(delta)
    df.insert(6, 'Δ DEU', delta_stolpec2)

    # change order of columns
    cols = ['Description', 'Category', 'Unit of measure', 'P-value', 'Scoring method', 'Higher = Better',
            'SVN', 'AUT', 'DEU', 'Δ AUT', 'Δ DEU']
    df = df[cols]
    df = df.sort_values('P-value', False)

    df.to_csv('A008W_data_table.csv')

    print(df)
    return data, df


create_df()