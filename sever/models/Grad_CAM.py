import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class GradCam(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_hooks()

    def register_hooks(self):
        layer = dict([*self.model.named_modules()])['encoder'][10].conv2
        layer.register_forward_hook(self.forward_hook)
        layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature_map = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, input_image, target_class=None):
        input_image.requires_grad = True
        
        # 모델에 입력 이미지를 통과시켜 forward 연산 수행
        outputs = self.model(input_image)

        # 클래스 인덱스가 지정되지 않은 경우, 가장 높은 확률의 클래스를 선택
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
            raw_output=outputs

        self.model.zero_grad()
        outputs[:, target_class].backward(retain_graph=True)

        # Grad-CAM 계산: gradient와 feature map을 이용하여 가중치 적용
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1).squeeze()

        # ReLU 적용 및 정규화
        cam = cam / cam.max()
        cam = F.relu(cam,inplace=False)

        return cam, F.softmax(raw_output)#,target_class