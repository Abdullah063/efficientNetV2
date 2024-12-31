from timm import create_model
from config import DEVICE

def get_model(num_classes):
    model = create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)