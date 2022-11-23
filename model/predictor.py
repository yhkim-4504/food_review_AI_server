import torch
import cv2
from torch.nn.functional import softmax
from torchvision import transforms

LABEL_TO_IDX = {'jeyuk_bokkeum': 0, 'kimbap': 1, 'omurice': 2, 'pork_cutlet': 3, 'ramen': 4, 'samgyetang': 5, 'tteokbokki': 6}


class FoodPredictor:
    def __init__(self, jit_path, img_shape=(256, 256), device='cuda:0'):
        self.device = device
        self.img_shape = tuple(img_shape)
        self.model = torch.jit.load(jit_path).to(self.device)
        self.model.eval()

        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def inference(self, image_array):
        # Preprocess image_array
        img = pad_to_square(image_array)
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)

        input_tensor = self.pre_transform(img)

        # Forward input
        with torch.no_grad():
            pred = self.model(input_tensor.unsqueeze(0).to(self.device)).cpu()
        pred_softmax = softmax(pred, dim=1)
        value, indices = torch.max(pred_softmax, dim=1)

        return {'index': indices.item(), 'prob': value.item()}

def pad_to_square(img):
    if img.shape[0]==img.shape[1]:
        return img
    
    length = max(img.shape)
    delta_w = length - img.shape[1]
    delta_h = length - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 255])
    
    return pad_img

if __name__ == '__main__':
    food_predictor = FoodPredictor('./Base2_chkpoint_jit_script.pt', (256, 256), 'cuda:0')
    image = cv2.imread('./ramen.jpg')
    result = food_predictor.inference(image)
    
    print(result)