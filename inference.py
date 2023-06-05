from dinov2.models.vision_transformer import vit_base, vit_giant2
import torch

from torchvision import transforms

from PIL import Image
import requests

# load model
def load_model(size="base"):

    if size == "base":
        model = vit_base(img_size=518, patch_size=14, init_values=1.0, ffn_layer = "mlp", block_chunks = 0)
        state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
                                                        map_location="cpu")
    elif size == "giant":
        model = vit_giant2(img_size=518, patch_size=14, init_values=1.0, ffn_layer = "swiglufused", block_chunks = 0)
        state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
                                                        map_location="cpu")

    # equip the model with weights
    model.load_state_dict(state_dict)

    return model



# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# preprocess image
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # these are RGB mean+std values
        std=[0.229, 0.224, 0.225]  # across a large photo dataset.
    )
])

pixel_values = transformations(image).unsqueeze(0)  # insert batch dimension

model = load_model(size="giant")
outputs = model.forward_features(pixel_values) 

for k,v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, v)