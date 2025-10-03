import streamlit as st
import torch
from model import Seq2Seq, EncoderBiLSTM, DecoderLSTM
import json
from utils.nmt_utils import translate_sentence, stoi, itos, PAD, BOS, EOS, UNK, device

st.set_page_config(page_title="Urdu → Roman Urdu NMT")

# Load vocab
with open("vocab/stoi.json", "r") as f:
    stoi = json.load(f)
with open("vocab/itos.json", "r") as f:
    itos = json.load(f)

# Load model
emb_dim = 128
enc_hidden = 256
dec_hidden = 256
vocab_size = len(stoi)

encoder = EncoderBiLSTM(vocab_size, emb_dim, enc_hidden).to(device)
decoder = DecoderLSTM(vocab_size, emb_dim, enc_hidden, dec_hidden).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load("model/best_char_seq2seq.pth", map_location=device))
model.eval()

st.title("Urdu → Roman Urdu Translator")
input_text = st.text_area("Enter Urdu text:")

if st.button("Translate"):
    if input_text.strip() != "":
        output = translate_sentence(model, input_text.strip())
        st.subheader("Roman Urdu Output:")
        st.write(output)
    else:
        st.warning("Please enter some Urdu text to translate.")
