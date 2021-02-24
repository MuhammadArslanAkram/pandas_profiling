import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.title("Exploratort Data Analysis using Pandas Profiling")

st.set_page_config("Pandas Profiling")
st.sidebar.markdown(""" **Developed by** [M.Arslan Akram](https://www.linkedin.com/in/arslanakram1/)
    """)
st.sidebar.markdown(""" **Source Code ** [Github](https://github.com/MuhammadArslanAkram/pandas_profiling)
    """)

with st.sidebar:
    uploaded_file=st.sidebar.file_uploader("Upload your CSV File","CSV")

if uploaded_file is not None:
    
    @st.cache
    def load_dataset():
        csv=pd.read_csv(uploaded_file)
        return csv

    df=load_dataset()
    pr = ProfileReport(df,explorative=True)
    st.header("Input Data Frame")
    st.write(df)
    st.write("---")
    st.header("Pandas Profiling")
    st_profile_report(pr)

else:
    st.warning("Waiting for File Uploading")
    if st.sidebar.button("Use IRIS Dataset"):
        
        @st.cache
        def load_iris():
            dl_iris=datasets.load_iris()
            df=pd.DataFrame(dl_iris.data,columns=dl_iris.feature_names)
            df['target'] = pd.Series(dl_iris.target)
            return df
        
        df = load_iris()
        
        pr = ProfileReport(df,explorative=True)
        st.header("Input Data Frame")
        st.write(df)
        st.write("---")
        st.header("Pandas Profiling")
        st_profile_report(pr)
