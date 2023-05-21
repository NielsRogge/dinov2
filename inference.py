from dinov2.models.vision_transformer import vit_base

model = vit_base()

for name, param in model.named_parameters():
    print(name, param.shape)