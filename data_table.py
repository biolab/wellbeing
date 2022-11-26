import pandas as pd
from Orange.data import Table
from feature_subset_selection import get_all_top_attributes



def create_df():
    data = Table("C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_A008.W_selected.pkl")
    d = pd.DataFrame(data.X)
    dm = pd.DataFrame(data.metas)

    df = pd.concat([dm, d], axis=1)

    meta_imena = data.metas_df.columns.to_list()
    imena = [attr.name for attr in data.domain.attributes]
    imena_kolon = meta_imena + imena
    df.columns = imena_kolon
    df = df.set_index('index') # vrstico dobim z indeksom drzave namesto s stevilko vrstice (npr df.loc['DEU'])

    top_factors = get_all_top_attributes(data)
    df = df[top_factors]
    df = df.loc[['SVN', 'AUT', 'DEU']].transpose()


    return data, df


#create_df()