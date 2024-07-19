import grpc
# from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel("172.17.0.2:9000")
try:
    grpc.channel_ready_future(channel).result(timeout=10)
    print("Connected to gRPC server")
except grpc.FutureTimeoutError:
    print("Failed to connect to gRPC server")
