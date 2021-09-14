
import streamlit as st

import speech_recognition as sr

r = sr.Recognizer()


st.title("Voice to Text")
st.write(" ")


with sr.AudioFile('common_voice_en_100034 (online-audio-converter.com).wav') as source:
  audio_text = r.record(source)


  try:
    text = r.recognize_google(audio_text)

    print(text)
    if st.checkbox('Check your text here'):
        st.write(text)


  except:
        st.write('Sorry...run again')


fe = st.radio(label='Keyword Extractions', options=['','Ngrams','Important Words'])
if (fe == 'Important Words'):
    import multi_rake
    from multi_rake import Rake
    rake = Rake()
    keywords = rake.apply(text)
    st.markdown(keywords[:10])
    
if (fe=='Ngrams'):
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    v1 = CountVectorizer(max_features=30, min_df=1,max_df=1.0,ngram_range=(3,3), stop_words='english')
    text = [text]
    x1 = v1.fit_transform(text).toarray()
    st.write(v1.get_feature_names())