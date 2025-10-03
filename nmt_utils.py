
import torch
import json
PAD, UNK, BOS, EOS = 0, 1, 2, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab (for Streamlit later)
with open("vocab/stoi.json", "r", encoding="utf-8") as f:
    stoi = json.load(f)
with open("vocab/itos.json", "r", encoding="utf-8") as f:
    itos = json.load(f)

def decode_ids(ids):
    tokens = [itos.get(int(x), "") for x in ids if int(x) not in (PAD, BOS)]
    return "".join(tokens).replace("</s>", "")

def translate_sentence(model, sentence, max_len=128):
    model.eval()
    ids = [stoi.get(ch, UNK) for ch in sentence]
    ids = [BOS] + ids + [EOS]
    src = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        encoder_outputs, _ = model.encoder(src)
        dec_h = torch.zeros(model.decoder.rnn.num_layers, 1,
                            model.decoder.rnn.hidden_size).to(device)
        dec_c = torch.zeros_like(dec_h)
        input_step = torch.tensor([[BOS]], dtype=torch.long).to(device)
        output_ids = []
        for _ in range(max_len):
            preds, (dec_h, dec_c) = model.decoder(input_step, (dec_h, dec_c), encoder_outputs)
            top1 = preds.argmax(1).item()
            if top1 == EOS:
                break
            output_ids.append(top1)
            input_step = torch.tensor([[top1]], dtype=torch.long).to(device)
    return decode_ids(output_ids)
