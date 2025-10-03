import streamlit as st
import torch
from model_utils import EncoderBiLSTM, DecoderLSTM, Seq2Seq, translate_sentence, vocab_size, device

# ----------------------------
# Load model
# ----------------------------
emb_dim = 128
enc_hidden = 256
dec_hidden = 256

encoder = EncoderBiLSTM(vocab_size, emb_dim, enc_hidden).to(device)
decoder = DecoderLSTM(vocab_size, emb_dim, enc_hidden, dec_hidden).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load("best_char_seq2seq.pth", map_location=device))
model.eval()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Urdu → Roman Urdu Transliteration")
st.write("Enter Urdu text below and get Roman Urdu transliteration:")

user_input = st.text_area("Urdu Input", height=150)

if st.button("Translate"):
    if user_input.strip():
        roman_output = translate_sentence(model, user_input.strip())
        st.success(roman_output)
    else:
        st.warning("Please enter some Urdu text!")
