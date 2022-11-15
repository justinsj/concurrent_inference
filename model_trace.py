import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet50.pt")

model = torchvision.models.inception_v3(pretrained=True)
model.eval()
example = torch.rand(1, 3, 299, 299)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("inception_v3.pt")

model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
model.eval()
example = torch.rand(1, 3, 518, 518)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("vit_h14_in1k.pt")


model = torchvision.models.vgg19_bn(pretrained=True)
model.eval()
example = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("vgg19_bn.pt")


model = torchvision.models.wide_resnet101_2(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("wide_resnet101_2.pt")


model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
model.eval()
example = torch.rand(1, 3, 384, 384)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("vit_b16_in1k.pt")