# train_binary.py
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------- LOAD CSV (message,label) --------
df = pd.read_csv("data.csv")
df = df.rename(columns={"message": "sms", "label": "is_transactional"})

texts = df["sms"].astype(str).values
labels = df["is_transactional"].astype(int).values  # 0/1

# -------- TOKENIZER --------
MAX_WORDS = 8000   # better for Indian data
MAXLEN = 80        # increase due to long Hindi/promo texts

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAXLEN)
y = labels

# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# -------- MODEL --------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_WORDS, 64, input_length=MAXLEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(1, activation="sigmoid")  # binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------- TRAIN --------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=6,
    batch_size=32
)

model.save("transaction_classifier.keras")
print("✅ Binary transactional model saved!")
