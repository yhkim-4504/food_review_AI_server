from __future__ import print_function

import logging

import grpc
import food_classification_pb2
import food_classification_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = food_classification_pb2_grpc.FoodAiStub(channel)
        response = stub.CheckGpuStatus(food_classification_pb2.Empty())
    print('response 1')
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = food_classification_pb2_grpc.FoodAiStub(channel)
        response = stub.PredictFoodImage(food_classification_pb2.FoodImage(image=b'asdasdasd'))
    #print("Greeter client received: " + response)
    print('response2')


if __name__ == '__main__':
    logging.basicConfig()
    run()
