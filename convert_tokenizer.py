import pickle
import json

# Load your tokenizer.pkl
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Save as JSON
tokenizer_json = tokenizer.to_json()

with open("tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

print("✅ tokenizer.json created")
