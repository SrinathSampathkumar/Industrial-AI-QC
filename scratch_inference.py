import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from anomalib.models import Patchcore

def main():
    model_path = Path("models/patchcore/bottle/model.pt")
    data = torch.load(model_path, weights_only=False)
    
    # Initialize the model using saved config
    model = Patchcore(
        backbone=data["backbone"],
        layers=data["layers"],
        coreset_sampling_ratio=data["coreset_sampling_ratio"],
        num_neighbors=data["num_neighbors"],
    )
    
    # Load state dict
    model.load_state_dict(data["model_state_dict"])
    
    # Crucial step: some parts of the memory bank might not be part of state_dict properly
    # Actually, anomalib models handle it. Let's try.
    model.eval()
    
    # Load an image
    image_path = list(Path("datasets/bottle/test/good").glob("*.png"))[0]
    img = Image.open(image_path).convert("RGB")
    
    # Typical transforms (assuming 256x256 resizing)
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        
    print(output)
    
if __name__ == "__main__":
    main()
