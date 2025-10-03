print("Hello, this is the Streamlit app placeholder!")

%%writefile /content/requirements.txt
streamlit
torch

import json
stoi = {"<pad>": 0, "a": 1, "b": 2}
itos = {0: "<pad>", 1: "a", 2: "b"}

with open("/content/stoi.json", "w") as f:
    json.dump(stoi, f)

with open("/content/itos.json", "w") as f:
    json.dump(itos, f)
