from ast import literal_eval

import streamlit as st
from annotated_text import annotated_text
from mudes.app.mudes_app import MUDESApp
import pandas as pd

en_base = MUDESApp("small", use_cuda=False)
en_large = MUDESApp("en-large", use_cuda=False)
multi_lingual_base = MUDESApp("multilingual-base", use_cuda=False)
multi_lingual_large = MUDESApp("multilingual-large", use_cuda=False)


def toxic_to_rgb(is_toxic: bool):
    if is_toxic:
        return "rgb(255, 0, 0)"
    else:
        return "rgb(211,211,211)"


def highlight(s, index, color='yellow'):
    if s.name == index:
        hl = f"background-color: {color}"
    else:
        hl = ""
    return [hl] * len(s)


def get_data(dataset_name):
    if dataset_name == "Civil Comments Dataset":
        data = pd.read_csv("mudes_ui/assets/data/tsd_trial.csv")
        data["spans"] = data.spans.apply(literal_eval)
        return data

    if dataset_name == "OLID":
        data = pd.read_csv('mudes_ui/assets/data/test_a_tweets.tsv', sep="\t")
        data = data.rename(columns={'tweet': 'text'})
        return data

    if dataset_name == "OGDT":
        data = pd.read_csv('mudes_ui/assets/data/offenseval-gr-test-v1.tsv', sep="\t")
        data = data.rename(columns={'tweet': 'text'})
        return data

    if dataset_name == "Danish":
        data = pd.read_csv("mudes_ui/assets/data/offenseval-da-test-v1.tsv", sep="\t")
        data = data.rename(columns={'tweet': 'text'})
        return data

    else:
        return None


def get_model(model_name):

    if model_name == "en-base":
        return en_base

    if model_name == "en-large":
        return en_large

    if model_name == "multilingual-base":
        return multi_lingual_base

    if model_name == "multilingual-large":
        return multi_lingual_large

    else:
        return None


# Keep the state between actions
@st.cache(allow_output_mutation=True)
def current_sentence_state():
    return {"index": 0}


def main():
    st.set_page_config(
        page_title='MUDES UI',
        initial_sidebar_state='expanded',
        layout='wide',
    )

    st.sidebar.title("MUDES")
    st.sidebar.markdown("Multilingual Detection of Offensive Spans")
    st.sidebar.markdown(
        "[code](https://github.com/TharinduDR/MUDES)"
    )

    st.sidebar.markdown("---")

    st.sidebar.header("Available Datasets")
    selected_dataset_name = st.sidebar.radio(
        'Select a dataset to use',
        ["Civil Comments Dataset", "OLID", "OGDT", "Danish"]
    )

    df = get_data(selected_dataset_name)

    st.sidebar.markdown("---")

    st.sidebar.header("Available Models")
    selected_model = st.sidebar.radio(
        'Select a pretrained model to use',
        ["en-base", "en-large", "multilingual-base", "multilingual-large"],
    )

    model = get_model(selected_model)

    st.header("Input a sentence")
    st.write(
        "Select a predefined sentence and/or edit the sentences"
    )

    current_state = current_sentence_state()

    col1, col2, *_ = st.beta_columns(12)
    with col1:
        previous_pressed = st.button('Previous')
    with col2:
        next_pressed = st.button('Next')

    if previous_pressed:
        current_state['index'] = max(0, current_state['index'] - 1)
    if next_pressed:
        current_state['index'] = min(len(df), current_state['index'] + 1)

    i = st.slider(
        'Scroll through dataset',
        min_value=0,
        max_value=len(df['text'].tolist()),
        value=current_state['index'],
    )
    current_state['index'] = i

    sentences = df['text'].tolist()
    sentence = sentences[i]

    with st.beta_expander('Preview sentences', expanded=False):
        first_idx = max(i - 7, 0)
        last_idx = min(first_idx + 15, len(sentences))
        df_w = df.iloc[first_idx:last_idx].style.apply(highlight, index=i, axis=1)
        st.dataframe(df_w, width=None, height=400)

    st.write('### Edit sentence')
    col1, col2 = st.beta_columns(2)
    with col1:
        sentence_text = st.text_area('Sentence', value=sentence)

    st.header('Toxic Spans')

    if selected_model == "en-base" or selected_model == "en-large":
        tokens = model.predict_tokens(sentence_text, language="en")

    else:
        tokens = model.predict_tokens(sentence_text, language="xx")

    predictions = st.beta_container()
    with predictions:
        text = [
            (token.text, "", toxic_to_rgb(token.is_toxic))
            for token in tokens
            ]
        st.write('Predicted Toxic spans in the sentence')
        annotated_text(*text)


if __name__ == "__main__":
    main()




