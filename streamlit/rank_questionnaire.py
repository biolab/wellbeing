import streamlit as st

st.title('Kje v Evropi bi najraje živeli?')
st.subheader('uvodni nagovor')

st.markdown('<a href="/countries_pairs_10" target="_self">Naslednja stran</a>', unsafe_allow_html=True)

def samo_po_sebi_namen():
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
            answers = []
            if submitted:
                results = ''
                for i in range(n_questions):
                    results += f'Vprašanje {i + 1}: {listek[i]}\n'
                    answers.append(listek[i])
                with open('results.txt', 'w', encoding='utf-16') as f:
                        f.write(results)

            return answers

    def get_demography():
        sex_list = ['Moški', 'Ženska']
        age_list = ['Do 25 let', '26-35 let', '36-45 let', '46-55 let', '56-65 let', 'Nad 65 let']
        region_list = ['Gorenjska', 'Goriška', 'Jugovzhodna', 'Koroška', 'Obalno-kraška', 'Osrednjeslovenska',
                        'Podravska', 'Pomurska', 'Posavska', 'Primorsko-notranjska', 'Savinjska', 'Zasavska']
        edu_list = ['Nedokončana osnovna šola', 'Osnovna šola', 'Dveletna ali triletna poklicna srednja šola',
                    'Štiriletna ali petletna srednja šola / Višja šola', 'Visoka šola, univerzitetna izobrazba (1. in 2. bolonjska stopnja)',
                    'Specializacija, znanstveni magisterij, doktorat']
        sex_listek = []
        age_listunja = []
        region_listmasina = []
        edu_lust = []

        with st.form('neko ime'):
            sex_quest = st.radio('Označite svoj spol:', sex_list, label_visibility="visible")
            sex_listek.append(sex_quest)

            age_quest = st.radio('Označite svojo starost:', age_list, label_visibility="visible")
            age_listunja.append(age_quest)

            edu_quest = st.radio('Označite svojo najvišjo stopnjo izobrazbe:', edu_list, label_visibility="visible")
            edu_lust.append(edu_quest)

            region_quest = st.radio('V katerem kraju živite?', region_list, label_visibility="visible")
            region_listmasina.append(region_quest)


            st.form_submit_button('Oddaj')


    odgovori = get_questions()
    get_demography()





