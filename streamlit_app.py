# Define the contents of requirements.txt
requirements = """
streamlit
torch
torchtext
transformers
sentencepiece
sacrebleu
"""

# Write to requirements.txt
with open("requirements.txt", "w") as f:
    f.write(requirements.strip())

print("✅ requirements.txt created successfully!")
