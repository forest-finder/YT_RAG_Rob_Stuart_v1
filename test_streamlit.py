import streamlit as st

st.title("🎥 Test Streamlit App")
st.write("Hello! This is a test to see if Streamlit works.")

name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

if st.button("Click me!"):
    st.balloons()
    st.success("Button clicked!")