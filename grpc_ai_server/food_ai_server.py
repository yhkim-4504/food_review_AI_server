import numpy as np
import cv2
import grpc
from concurrent import futures
from grpc_ai_server import food_classification_pb2
from grpc_ai_server import food_classification_pb2_grpc
from model.predictor import FoodPredictor


class FoodAiServicer(food_classification_pb2_grpc.FoodAiServicer):
    print('Food Classification Model loading...', end='')
    food_predictor = FoodPredictor('./model/Base2_chkpoint_jit_script.pt', (256, 256), 'cuda:0')
    print('Done.')

    def CheckGpuStatus(self, request, context):
        print('CheckGpu grpc')
        return food_classification_pb2.GpuStatus(status=True)
    
    def PredictFoodImage(self, request, context):
        print('PredictFoodImage grpc')
        image = cv2.imdecode(np.frombuffer(request.image, np.uint8), cv2.IMREAD_COLOR)
        result = self.food_predictor.inference(image)

        return food_classification_pb2.PredictionResult(food_type=result['index'], probability=result['prob'])



def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    food_classification_pb2_grpc.add_FoodAiServicer_to_server(FoodAiServicer(), server)
    server.add_insecure_port('127.0.0.1:' + port)  # localhost
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()
