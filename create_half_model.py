import torch
import torchvision
import torch.nn as nn

model_name = 'vit_h14_in1k'
model = torch.jit.load(model_name+'.pt')


def load_modules(model, max_depth):
    modules = []
    load_list_of_modules(model, modules, max_depth)
    return modules
def load_list_of_modules(module, list_of_modules, remaining_depth = 0):
    if remaining_depth == 0:
        return
    for child in module.children():
        if len(list(child.children())) == 0 or remaining_depth == 1:
            list_of_modules.append(child)
        load_list_of_modules(child, list_of_modules, remaining_depth - 1)




# @torch.jit.script
def get_next_input(module, module_idx, total_modules, example):
    example = module(example)

    if module_idx == 1:
        example = example.permute(0, 2, 3, 1)
        example = example.reshape(1369, 1, 1280)
    
    if module_idx == total_modules - 2:
        print("EXAMPLE SHAPE:", example.shape)
        example = example[0]

    if module_idx == total_modules - 1:
        print("OTHER EXAMPLE SHAPE:", example.shape)
        example = example.reshape(1, 1000)
    return example


# model_half = torch.nn.Sequential(*first_half)

class Model(torch.nn.Module):
    def __init__(self, modules, total_modules):
        super(Model, self).__init__()
        # Register the modules
        self.modules = modules
        self.total_modules = total_modules
        for i, l in enumerate(self.modules):
            self.add_module(str(i) + 's', l)
    
    def forward(self, x):
        for module_idx, module in enumerate(self.modules):
            # module(x)
            x = get_next_input(module, module_idx, self.total_modules, x)
        return x
# model_half = Model(first_half)

# @torch.jit.script
# def model_half(example, idx, modules):
#     module = modules[idx]
#     example = get_next_input(module, module_idx, len(modules), example)
#     return example

# def the_model(example):
#     for idx, module in enumerate(modules):
#         example = model_half(example, idx, modules)
#     return example
    
# model_half.eval()

modules = load_modules(model, 3)

first_half = modules[:len(modules)//2]
print("len(first_half):", len(first_half))

model_half = Model(first_half, len(modules))

example = torch.rand(1, 3, 518, 518)

# test = model_half(example)
traced_model = torch.jit.trace(model_half, example)
path = model_name + '_half.pt'
traced_model.save(path)

traced_model = torch.jit.load(path)
traced_model.eval()
example = torch.rand(1, 3, 518, 518)
traced_model(example)
