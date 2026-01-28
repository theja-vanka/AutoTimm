"""Explore available timm backbones and inspect model properties."""

import autotimm

# List all available backbones (1000+)
all_models = autotimm.list_backbones()
print(f"Total available backbones: {len(all_models)}")

# Filter by architecture family
resnets = autotimm.list_backbones("resnet*")
print(f"\nResNets: {len(resnets)}")
for m in resnets[:5]:
    print(f"  {m}")

efficientnets = autotimm.list_backbones("efficientnet*")
print(f"\nEfficientNets: {len(efficientnets)}")
for m in efficientnets[:5]:
    print(f"  {m}")

vits = autotimm.list_backbones("vit_*")
print(f"\nVision Transformers: {len(vits)}")
for m in vits[:5]:
    print(f"  {m}")

convnexts = autotimm.list_backbones("convnext*")
print(f"\nConvNeXts: {len(convnexts)}")
for m in convnexts[:5]:
    print(f"  {m}")

# Only models with pretrained weights
pretrained = autotimm.list_backbones("*mobilenet*", pretrained_only=True)
print(f"\nPretrained MobileNets: {len(pretrained)}")
for m in pretrained[:5]:
    print(f"  {m}")

# Create a backbone and inspect it
backbone = autotimm.create_backbone("resnet50")
print(f"\nresnet50 output features: {backbone.num_features}")
print(
    f"resnet50 total params: {autotimm.count_parameters(backbone, trainable_only=False):,}"
)
