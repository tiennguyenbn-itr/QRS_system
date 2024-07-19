import time
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_data
# tf.config.set_visible_devices([], 'GPU')

# Load the SavedModel
model = tf.saved_model.load("model_new_2/1")

# Get the default serving signature
infer = model.signatures["channels"]

# Prepare input data
MITDB_DIR = '/home/tien/Documents/ITR/mit-bih-arrhythmia-database-1.0.0/'
test_data, _ = preprocess_data(MITDB_DIR+'100.hea')
x_test = test_data[:1024,:,:]

# Convert to a TensorFlow tensor if necessary
input_tensor = tf.convert_to_tensor(x_test)

# Run inference
start = time.time()
outputs = infer(input_tensor)
end = time.time()
print(f"Time: {end-start} seconds")

# Print output
for key, value in outputs.items():
    print(f"{key}: {value.numpy()}")
