import itertools
import streamlit as st
import translators.server as tss
from Orange.data import Table
from random import shuffle


st.title('Kje v Evropi bi najraje živeli?')
st.subheader('uvodni nagovor')


def get_translate():
    data = Table('C:\\Users\irisc\Documents\FRI\\blaginja\FRI-blaginja\SEI_krajsi_ranking_selected.pkl')
    cntrs_list = data.metas_df['Country name'].tolist()
    str_cntrs = ' | '.join(cntrs_list)
    str_translated = tss.google(str_cntrs, from_language='en', to_language='sl')
    str_translated = str_translated.split(' |')
    for i in range(len(str_translated)):
        if str_translated[i] == 'Makedonija, Republika':
            str_translated[i] = 'Severna Makedonija'
    return str_translated


def get_answer_pairs():
    create_answers = get_translate()
    unique_combinations = []
    combinations = itertools.combinations(create_answers, 2)
    for comb in combinations:
        unique_combinations.append(comb)
    shuffle(unique_combinations)
    return unique_combinations

def get_questions():
    all_questions = get_answer_pairs()
    n_questions = 50
    listek = []
    with st.form('neko ime mora bit'):
        for i in range(n_questions):
            if i == 0:
                question = st.radio('Raje bi živel v:', all_questions[0], horizontal=True)
            elif i == 12:
                question = st.radio('Raje bi živel v:', all_questions[0], key='testni osebek',
                                    horizontal=True, label_visibility="collapsed")
            else:
                question = st.radio('Raje bi živel v:', all_questions[i], horizontal=True, label_visibility="collapsed")
            listek.append(question)

        submitted = st.form_submit_button('Naprej')
        if submitted:
            results = ''
            for i in range(n_questions):
                results += f'Vprašanje {i + 1}: {listek[i]}\n'
            with open('results.txt', 'w', encoding='utf-16') as f:
                    f.write(results)


get_questions()