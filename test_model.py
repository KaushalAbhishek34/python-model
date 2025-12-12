# test_binary_model.py
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Load AI Model + Tokenizer ----
model = tf.keras.models.load_model("transaction_classifier.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 80  # Must match training


AMOUNT_REGEX = r"(?:INR|Rs\.?|Rs|₹)\s?([0-9,]+(?:\.[0-9]+)?)|(?:debited|paid|spent|transfer(?:red)?|trf|deducted|withdrawn)\s+(?:by\s+)?([0-9,]+(?:\.[0-9]+)?)"


COUNTERPARTY_PATTERNS = [
    r"to\s+([A-Za-z ]+)",                   # trf to NAME
    r"from\s+([A-Za-z ]+)",                 # transfer from NAME
    r"towards\s+([A-Za-z ]+)",              # bill towards NAME
    r"at\s+([A-Za-z ]+)",                   # POS at NAME
]

BANK_FILTER_WORDS = [
    "sbi", "bank", "hdfc", "icici", "axis", 
    "kotak", "pnb", "bob", "boi", "ref", "refno",
    "a/c", "account", "ac", "acc","no", "refno", "ref", "-", "online", "or"
]

# ---- Extract amount ----
def extract_amount(text):
    m = re.search(AMOUNT_REGEX, text, flags=re.IGNORECASE)
    if not m:
        return None
    amt = m.group(1) or m.group(2)
    try:
        return float(amt.replace(",", ""))
    except:
        return None

# ---- Extract credit/debit type ----
def extract_type(text):
    t = text.lower()
    if "credited" in t or "credit" in t or "received" in t or "deposit" in t:
        return "credit"
    if "debited" in t or "debit" in t or "spent" in t or "transfer" in t or "trf" in t or "sent" in t:
        return "debit"
    return None


def extract_counterparty(text):
    t = text.lower()

    for pattern in COUNTERPARTY_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()

            # remove extra words like REF, A/C, SBI etc.
            words = name.split()
            clean_words = [
                w for w in words 
                if w.lower() not in BANK_FILTER_WORDS and not w.isdigit()
            ]

            clean_name = " ".join(clean_words).strip()
            return clean_name if clean_name else None

    return None


# ---- Extract cleaned description ----
def build_description(counterparty, amount, txn_type):
    if not amount or not txn_type:
        return None

    if counterparty:
        return f"{txn_type.title()} of Rs {amount} with {counterparty}"

    return f"{txn_type.title()} of Rs {amount}"


# ------------------ FINAL PREDICTION LOGIC ------------------

def predict_sms(text):
    # AI PREDICT
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    prob = float(model.predict(padded)[0][0])
    is_txn = prob >= 0.50

    if not is_txn:
        return {
            "sms": text,
            "ai_probability": prob,
            "is_transactional": False,
            "amount": None,
            "type": None,
            "description": None
        }

    # EXTRACT DETAILS
    amount = extract_amount(text)
    txn_type = extract_type(text)
    counterparty = extract_counterparty(text)
    description = build_description(counterparty, amount, txn_type)

    return {
        "sms": text,
        "ai_probability": prob,
        "is_transactional": True,
        "amount": amount,
        "type": txn_type,
        "description": description
    }



# ------------------ TEST MESSAGES ------------------

test_messages = [
    "Dear SBI User, your A/c X9376-credited by Rs.5000 on 05Dec25 transfer from Sidharth Kaushal Ref No 570540025982 -SBI",
    "Dear SBI User, your A/c X9376-credited by Rs.300 on 23Oct25 transfer from SHREYANSH  PARIHAR Ref No 529668389575 -SBI",
    "Dear UPI user A/C X9376 debited by 224.33 on date 05Nov25 trf to Zomato Online Or Refno 530996557064 If not u? call-1800111109 for other services-18001234-SBI",
    "Dear SBI User, your A/c X9376-credited by Rs.500 on 02Dec25 transfer from RAJESH KUMAR S O TILAK RAJ Ref No 533608121751 -SBI",
    "बधाई हो! आप Google Gemini के Rs. 35100 मूल्य के 18 महीने के मुफ़्त प्रो प्लान के लिए पात्र हैं। फ़ोटो के लिए Nano Banana, वीडियो के लिए Veo 3.1 और 2 TB क्लाउड स्टोरेज का आनंद लें। अभी क्लेम  करें www.jio.com/r/ChYTHwGjb । Jio नियम व शर्तें लागू।",
    "Hi Your login OTP for Burger King App is 0331.",
    "Subway Alert! Get Flat 10% OFF across our wide range of sandwiches, salads & wraps when you visit next. Code: 156dzw9om Valid till 22 Nov. TnC",
    "Dear customer, the clock is ticking, Get up to Rs 250 off on all OZiva products.Don't wait, 48 hrs only! Use Code STEAL Shop Now https://oziva.me/iOZIVA/TJOUt",
    "Dear UPI user A/C X9376 debited by 10.0 on date 10Nov25 trf to ANKIT SAINI S O Refno 568077181465 If not u? call-1800111109 for other services-18001234-SBI",
    "Dear UPI user A/C X9376 debited by 120.0 on date 04Nov25 trf to PRINCE PRASHAR Refno 530861143072 If not u? call-1800111109 for other services-18001234-SBI"
]

for m in test_messages:
    print("------------------------------------------------")
    print(predict_sms(m))
