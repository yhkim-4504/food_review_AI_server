import numpy as np
import cv2
import grpc
import threading
from concurrent import futures
from . import food_classification_pb2
from . import food_classification_pb2_grpc
from model.predictor import FoodPredictor
from collections import deque
from time import sleep
from torch import cuda
from datetime import datetime

QUEUE_WAITING_TIME = 0.001

class FoodAiServicer(food_classification_pb2_grpc.FoodAiServicer):
    print('Food Classification Model loading...', end='')
    food_predictor = FoodPredictor('./model/Base2_chkpoint_jit_script.pt', (256, 256), 'cuda:0')
    waiting_queue = deque()
    process_id = 0
    thread_lock = threading.Lock()
    print('Done.')

    def CheckGpuStatus(self, request, context):
        print('CheckGpu grpc')

        if cuda.is_available():
            status = True
        else:
            status = False

        return food_classification_pb2.GpuStatus(status=status)
    
    def PredictFoodImage(self, request, context):
        with self.thread_lock:
            current_process_id = self.process_id

            print(datetime.now(), 'PredictFoodImage grpc', current_process_id)
            self.waiting_queue.append(current_process_id)
            self.process_id += 1
        

        while self.waiting_queue and (self.waiting_queue[0] != current_process_id):
            sleep(QUEUE_WAITING_TIME)

        image = cv2.imdecode(np.frombuffer(request.image, np.uint8), cv2.IMREAD_COLOR)
        result = self.food_predictor.inference(image)

        with self.thread_lock:
            self.waiting_queue.popleft()
        print(datetime.now(), current_process_id, 'Done', self.waiting_queue)

        return food_classification_pb2.PredictionResult(food_type=result['index'], probability=result['prob'])



def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    food_classification_pb2_grpc.add_FoodAiServicer_to_server(FoodAiServicer(), server)
    #server.add_insecure_port('0.0.0.0:' + port)  # localhost
    server.add_insecure_port('[::]:' + port)  # localhost
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

serve()