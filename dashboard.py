import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pickle
import librosa
from speech_model_infer import Speech_Model_Infer

import os
import sys


class Dashboard:

    def __init__(self, title: str) -> None:
        self.title = title

        self.model_infer = self.init_speech_model_infer()

    @st.cache()
    def init_speech_model_infer(self):
        model_infer = Speech_Model_Infer()
        return model_infer

    def render_siderbar(self, pages, select_label):
        st.sidebar.markdown("# Pages")
        self.page = st.sidebar.selectbox(f'{select_label}', pages)

    def predict(self, test_df: pd.DataFrame, model):
        pass

    def render(self):
        st.title(f"Welcome To {self.title}")
        self.render_siderbar([
            'Upload audio file', "Manual",
        ], "select method: ")

        if (self.page == "Upload audio file"):
            wav_file = st.file_uploader("Upload wav file", type=['wav'])
            if (wav_file):
                st.audio(wav_file, format='audio/wav')

                if st.button('Predict'):
                    st.write(wav_file)
                    pred = self.model_infer.predict(wav_file)
                    if pred:
                        st.write(pred)
                    else:
                        st.write("Could not predict")

        elif (self.page == "Manual"):
            st.markdown("### Record")
            # self.render_overview()


if __name__ == "__main__":
    dashboard = Dashboard("Amharic speech to text")
    dashboard.render()
