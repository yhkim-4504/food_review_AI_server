from __future__ import print_function


import grpc
from grpc_ai_server import food_classification_pb2
from grpc_ai_server import food_classification_pb2_grpc


def run():
    with open('./model/ramen.jpg', 'rb') as f:
        image = f.read()

    for _ in range(100):
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = food_classification_pb2_grpc.FoodAiStub(channel)
            response = stub.PredictFoodImage(food_classification_pb2.FoodImage(image=image))
            print(response)


if __name__ == '__main__':
    run()
