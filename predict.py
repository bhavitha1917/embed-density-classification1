import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Model
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(
        model.fc.in_features,
        512
    ),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(
        512,
        4
    )
)

# Load checkpoint
checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])


def predict_image(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")

    x = transform(img)

    x = x.unsqueeze(0)

    x = x.to(device)

    with torch.no_grad():

        output = model(x)

        pred = torch.argmax(
            output,
            dim=1
        )

    return int(pred.item())


# Example

if __name__ == "__main__":

    image_path = "test/test_image.png"

    prediction = predict_image(
        image_path
    )

    print(
        f"Predicted Density Class: {prediction}"
    )