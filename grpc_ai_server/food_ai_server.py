# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc
import food_classification_pb2
import food_classification_pb2_grpc

from time import sleep
class Greeter(food_classification_pb2_grpc.FoodAiServicer):
    def CheckGpuStatus(self, request, context):
        print('CheckGpu grpc')
        sleep(3)
        return food_classification_pb2.GpuStatus(status=True)
    
    def PredictFoodImage(self, request, context):
        print('PredictFoodImage grpc')
        sleep(3)
        return food_classification_pb2.PredictionResult(food_type=1, probability=0.97)



def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    food_classification_pb2_grpc.add_FoodAiServicer_to_server(Greeter(), server)
    server.add_insecure_port('127.0.0.1:' + port)  # localhost
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
