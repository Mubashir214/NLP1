print("Hello, this is the Streamlit app placeholder!")

# Create requirements.txt
requirements = ["streamlit", "torch"]
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

# Create stoi.json and itos.json
import json

stoi = {"<pad>": 0, "a": 1, "b": 2}
itos = {0: "<pad>", 1: "a", 2: "b"}

with open("stoi.json", "w") as f:
    json.dump(stoi, f)

with open("itos.json", "w") as f:
    json.dump(itos, f)

print("✅ requirements.txt, stoi.json, and itos.json created successfully!")
