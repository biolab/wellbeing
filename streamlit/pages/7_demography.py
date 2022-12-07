import streamlit as st

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

get_demography()