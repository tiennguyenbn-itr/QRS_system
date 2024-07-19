from preprocessing import preprocess_data
import tensorflow as tf
import grpc
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Set visible devices to CPU only
# tf.config.set_visible_devices([], 'GPU')

# Enable GPU
physical_devices = tf.config.list_physical_devices('GPU')

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
    # print(imgs.shape)
    tensor_proto = tf.make_tensor_proto(imgs, dtype=np.float32, shape=imgs.shape)
    # print(tensor_proto)
    request.inputs["input"].CopyFrom(tensor_proto)

    try:
        result = stub.Predict(request, 30.0)
        result = result.outputs["prediction"]
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Preprocess data
MITDB_DIR = '/home/tien/Documents/ITR/mit-bih-arrhythmia-database-1.0.0/'
test_data, _ = preprocess_data(MITDB_DIR + '100.hea')
y_pred = grpc_infer(test_data[:3,:,:])

print(y_pred)
