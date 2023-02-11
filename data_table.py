import pandas as pd
from Orange.data import Table
from fss import get_all_top_attributes
from p_values import get_p_values_for_top_factors


def create_df():
    filepath = r'C:\\Users\\irisc\\Documents\\FRI\\blaginja\\FRI-blaginja\\input data\\ranking_survey.pkl'
    data = Table(filepath)
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


    df = df.dropna()                                                    # remove NaN values
    att_names = df.index.to_list()                                      # dobim seznam imen vseh atributov
    list_p_values = [p_values[att_name][0] for att_name in att_names]   # dobim seznam vseh p-values
    df.insert(2, 'P-value', list_p_values)                              # p-valuese vstavim v df na mesto 0
    list_method_names = [p_values[att_name][1] for att_name in att_names]
    df.insert(3, 'Scoring method', list_method_names)                   # imena metod vstavim na df na mesto 1
    print(df)

    df2 = pd.read_csv(r'C:\Users\irisc\Documents\FRI\blaginja\FRI-blaginja\input data\indicators_shorter.csv')
    df2 = df2.set_index('index')
    types = [df2.loc[att_name, 'type'] for att_name in att_names]
    df.insert(1, 'Category', types)
    descriptions = [df2.loc[att_name, 'description'] for att_name in att_names]
    df.insert(0, 'Description', descriptions)

    # import additional file, which contains information about the units of measure
    # CAREFUL! delimiter for file with measures units of rankings must be set to ';'
    # df3 = pd.read_csv(r'C:\Users\irisc\Documents\FRI\blaginja\input data\_measure_units.csv')
    df3 = pd.read_csv(r'C:\Users\irisc\Documents\FRI\blaginja\FRI-blaginja\input data\ranking_measure_units.csv', delimiter=';')
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
        x = df.loc[att_name, 'Higher = Better']
        if x == 0:
            delta = delta*(-1)
        delta_stolpec1.append(delta)
    df.insert(3, 'Δ AUT', delta_stolpec1)

    # get delta SVN - DEU
    delta_stolpec2 = []
    for att_name in att_names:
        delta = (df.loc[att_name, 'SVN'] - df.loc[att_name, 'DEU']) / max(df.loc[att_name, ['SVN', 'AUT', 'DEU']])
        x = df.loc[att_name, 'Higher = Better']
        if x == 0:
            delta = delta*(-1)
        delta_stolpec2.append(delta)
    df.insert(6, 'Δ DEU', delta_stolpec2)

    # change order of columns
    cols = ['Description', 'Category', 'Unit of measure', 'P-value', 'Scoring method', 'Higher = Better',
            'SVN', 'AUT', 'DEU', 'Δ AUT', 'Δ DEU']
    df = df[cols]
    df = df.sort_values('P-value', False)


    if filepath.endswith('A008.W.pkl'):
        df.loc['SP.DYN.CDRT.IN', 'Scoring method'] = 'L/F'
    elif filepath.endswith('A170.W.pkl'):
        df.loc['ilc.mdes01', 'Scoring method'] = 'L/F'
        df.loc['ilc.mddd17', 'Scoring method'] = 'L/F'
    elif filepath.endswith('SWB.LS.pkl'):
        df.loc['ilc.mdho07', 'Scoring method'] = 'L/F'
    elif filepath.endswith('ranking_survey.pkl'):
        df.loc['SP.DYN.LE00.IN', 'Scoring method'] = 'L/F'
        df.loc['SP.DYN.LE00.FE.IN', 'Scoring method'] = 'L/F'
        df.loc['SH.DYN.NCOM.ZS', 'Scoring method'] = 'L/F'
        df.loc['SH.DYN.NCOM.MA.ZS', 'Scoring method'] = 'L/F'
    else:
        raise ValueError('napacno ime kolega')

    print(df)

    ime_fajla = filepath.split('\\')[-1][:-4] # extract name of input file from filepath, remove '.pkl' from the end.
    df.to_csv(output_name)

    return data, df

def google(ime_datoteke):
    google = True
    if google:
        with open(f'{ime_datoteke}.csv') as f:
            tabela_str = f.read()
        tabela_str = tabela_str.replace(',', ':').replace('.', ',')
        with open(f'{ime_datoteke}.txt', 'w') as f:
            f.write(tabela_str)
        print(f"za google zapisano v {ime_datoteke}.txt")


create_df()