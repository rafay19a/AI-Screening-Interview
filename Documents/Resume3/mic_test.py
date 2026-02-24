import streamlit as st

st.title("Mic Test")
audio = st.audio_input("Speak")

if audio:
    st.success("Audio captured!")
