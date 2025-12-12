import tensorflow as tf

# Load keras model
model = tf.keras.models.load_model("transaction_classifier.keras", compile=False)

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # <-- required for LSTM models
]

converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()


# Save .tflite file
with open("sms_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ sms_model.tflite created successfully")
