from preprocessing import preprocess_data
import tensorflow as tf
import grpc
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from define import *
from util import evaluate
from ec57_test import ec57_eval
from make_data import get_record_raw
import os

# Establish gRPC channel
channel = grpc.insecure_channel("172.17.0.2:9000")
grpc.channel_ready_future(channel).result(timeout=10)
print("Connected to gRPC server")

# Create gRPC stub
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create PredictRequest
request = predict_pb2.PredictRequest()
request.model_spec.name = "qrs"
request.model_spec.signature_name = "channels"

def grpc_infer(imgs):
    tensor_proto = tf.make_tensor_proto(imgs, dtype=tf.float32, shape=imgs.shape)
    request.inputs["input"].CopyFrom(tensor_proto)

    try:
        result = stub.Predict(request, 30.0)
        result = result.outputs["prediction"]
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Define constants
step = 100000
dataset = MITDB_DIR

# Create directories
result_dir = RESULT_DIR_IF + 'ec57/' + "mit-bih-arrhythmia-database-1.0.0" + '/'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

ann_dir = TEMP_DIR_IF + "mit-bih-arrhythmia-database-1.0.0" + '/'
if not os.path.isdir(ann_dir):
    os.makedirs(ann_dir)

# Infer with TensorFlow Serving
for file in get_record_raw(dataset):
    name = file.split('/')[-1][:-4]
    if name in ['104', '102', '107', '217', 'bw', 'em', 'ma']:
        continue
    print(file)
    if os.path.exists(ann_dir+name+'.atr') and os.path.exists(ann_dir+name+'.pred'):
        pass
    else:
        test_data, _ = preprocess_data(file)
        y_pred = np.zeros((0, 2))
        for i in range(0, len(test_data), step):
            prediction = tf.make_ndarray(grpc_infer(test_data[i:i+step,:,:]))
            prediction = np.rint(prediction)
            y_pred = np.concatenate((y_pred, prediction), axis=0)
            print(f"Processed {i} samples")

        np.savetxt(TEMP_DIR_IF + 'pred.txt', y_pred, fmt='%d\t')
        evaluate(file.split('/')[-1][:-4], y_pred, MITDB_DIR, True)

# Evaluate by EC57
ec57_eval(result_dir, ann_dir, 'atr', 'atr', 'pred', None)