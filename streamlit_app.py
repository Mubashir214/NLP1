import streamlit as st
import json

st.write("Hello, this is the Streamlit app placeholder!")

# Example dictionaries
stoi = {"<pad>": 0, "a": 1, "b": 2}
itos = {0: "<pad>", 1: "a", 2: "b"}

# Save to JSON files in the current directory
with open("stoi.json", "w") as f:
    json.dump(stoi, f)

with open("itos.json", "w") as f:
    json.dump(itos, f)
