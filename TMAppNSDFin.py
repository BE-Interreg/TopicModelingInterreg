import streamlit as st
import pyLDAvis.gensim_models
import pyLDAvis
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Funktion zum Laden der Daten
def load_data(file):
    data = pd.read_csv(file)
    return data

# Funktion zum Laden der Stopwords aus einer hochgeladenen Datei
def load_stopwords(file):
    stopwords = list(file.read().decode("utf-8").splitlines())
    return stopwords

# Funktion zur Textbereinigung mit CountVectorizer und N-Gram-Range
def preprocess_texts(data, columns, stopwords, ngram_range):
    texts = data[columns].fillna('').agg(' '.join, axis=1)
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=ngram_range)
    transformed_texts = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    cleaned_texts = [' '.join([terms[i] for i in transformed_text.nonzero()[1]]) for transformed_text in transformed_texts]
    return cleaned_texts

# Funktion zur stabilen Benennung der Topics mit linearer Nummerierung
def get_topic_names(lda_model, topic_order, num_words=3):
    topic_names = []
    for idx, topic_id in enumerate(topic_order):
        # Hole die wichtigsten Wörter für jedes Topic aus dem Modell
        top_words = [word for word, _ in lda_model.show_topic(topic_id - 1, topn=num_words)]
        top_words_str = ", ".join(top_words)
        # Weise dem Topic eine lineare Nummerierung zu (statt der Modell-ID)
        topic_names.append(f"Topic {idx + 1}: {top_words_str}")
    return topic_names

# Streamlit-App-Konfiguration
st.set_page_config(layout="wide")
st.title("Interaktive Topic Modelling App mit stabilisierten Ergebnissen")

# Hochladen der CSV-Datei
uploaded_file = st.file_uploader("Wähle eine CSV-Datei aus", type="csv")

# Hochladen der Stopwords-Datei
stopwords_file = st.file_uploader("Lade eine Stopword-Liste hoch (.txt)", type="txt")

# Stopwords aus Datei laden, falls vorhanden
stopwords = load_stopwords(stopwords_file) if stopwords_file else None

if uploaded_file:
    # Daten laden
    data = load_data(uploaded_file)

    # Spaltenauswahl für die Analyse
    columns = st.multiselect("Wähle Spalten für das Topic Modelling", data.columns)

    # Parameter für Topic Modelling
    num_topics = st.slider("Anzahl der Themen", min_value=2, max_value=20, value=5, step=1)
    num_terms = st.slider("Anzahl der Wörter pro Thema", min_value=5, max_value=30, value=10, step=1)
    lambda_value = st.slider("Lambda (Relevanzmetriks)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)

    # N-Gram Range Auswahl
    ngram_min = st.slider("Minimale N-Gram Größe", 1, 3, 1)
    ngram_max = st.slider("Maximale N-Gram Größe", 1, 3, 2)
    ngram_range = (ngram_min, ngram_max)

    # Seed festlegen
    np.random.seed(42)

    # Texte bereinigen und das gewählte N-Gram-Range und Stopwords anwenden
    if columns:
        texts = preprocess_texts(data, columns, stopwords, ngram_range)
        dictionary = corpora.Dictionary([text.split() for text in texts])
        corpus = [dictionary.doc2bow(text.split()) for text in texts]

        # Stabilisiere das LDA-Modell durch festgelegten Seed und Mehrfachdurchläufe
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=42)

        # Visualisierung mit pyLDAvis
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, R=num_terms)

        # Namen der Topics basierend auf der pyLDAvis-Sortierung generieren
        topic_order = vis.to_dict()['topic.order']
        topic_names = get_topic_names(lda_model, topic_order, num_words=3)

        # pyLDAvis-Visualisierung anzeigen
        pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(pyLDAvis_html, width=1200, height=800)

        # Abstand einfügen, um Topic-Namen weiter unten anzuzeigen
        st.markdown("---")
        st.subheader("Automatisch generierte Namen für Topics:")
        for name in topic_names:
            st.write(name)

        # Download der HTML-Datei ermöglichen
        st.markdown("### Download der Visualisierung:")
        st.download_button(
            label="Download HTML",
            data=pyLDAvis_html,
            file_name="topic_modeling.html",
            mime="text/html"
        )