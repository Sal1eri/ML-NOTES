from vision.models.cnn.model import SmallCifarCNN
from vision.models.vit.model import TinyVisionTransformer


def build_model(name: str, num_classes: int = 10):
    name = name.lower()
    if name == "cnn":
        return SmallCifarCNN(num_classes=num_classes)
    if name == "vit":
        return TinyVisionTransformer(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}. Choose from: cnn, vit")

