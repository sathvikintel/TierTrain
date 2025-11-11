import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time 
import os 

# ======= Argument Parser =======
parser = argparse.ArgumentParser(description="Train ResNet on CIFAR datasets")
parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100'], required=True, help="Dataset: cifar10 or cifar100")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--resnet_layers", type=int, choices=[18, 34, 50, 152], default=18, help="ResNet variant")
args = parser.parse_args()

# ======= Dataset Loading =======
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

## TIER TRAIN CHANGES
dict_list = []
fwd_id = 1
bwd_id = 1
initial_time = 0

with open("log_files/utc_start_time.txt", 'r') as file_handle:
    for line in file_handle:
        initial_time = float(str(line))
    file_handle.close()

with open("log_files/log_lines.txt", 'w') as file:
    pass

def add_marker(desc, color):
    dict_list.append({"time": time.time() - initial_time, "line": f"{desc}", "color": color})

# ======= Hooks =======
def forward_pre_hook(module, input):
    global fwd_id
    print(f"FWD ID = {fwd_id}")
    with open("log_files/obj_dump.log", 'a') as file:
        file.write(f"FWD ID : {fwd_id}\n")
    fwd_id += 1
    add_marker("fwd_layer", "red")

# Define a hook function that gets called during backward pass
def backward_hook(module, grad_output, grad_input):
    global bwd_id
    print(f"BWD ID = {bwd_id}")
    with open("log_files/obj_dump.log", 'a') as file:
        file.write(f"BWD ID: {bwd_id}\n")
    bwd_id += 1
    add_marker("bwd_layer", "blue")

def print_tensor_size_in_gb(tensor):
    # Get the total number of elements in the tensor
    num_elements = tensor.numel()

    # Get the size of each element in bytes
    element_size_bytes = tensor.element_size()

    # Calculate the total size of the tensor in bytes
    total_size_bytes = num_elements * element_size_bytes

    # Convert bytes to gigabytes (1 GB = 1024^3 bytes)
    total_size_gb = total_size_bytes / (1024 ** 3)

    # Print the size of the tensor in gigabytes
    print("Size of the tensor:", total_size_gb, "GB \n")

def bytes_to_gb(size_in_bytes):
    gb_size = size_in_bytes / (1024 ** 3)
    return gb_size

def compute_tensor_hash(tensor):
    # Convert to float tensor
    float_tensor = torch.tensor(tensor, dtype=torch.float32)
    
    # Convert to numpy array and get bytes
    tensor_bytes = float_tensor.numpy().tobytes()
    
    # Compute and return hash
    return hash(tensor_bytes)

def get_tensor_size_in_gb(tensor):
    if tensor is None:
        return 0.0
    # Calculate the number of bytes in the tensor
    num_bytes = tensor.element_size() * tensor.numel()

    # Convert bytes to gigabytes
    size_in_gb = num_bytes / (1024 ** 3)

    return size_in_gb

def get_page_aligned_address(virtual_address):
    # Calculate the page size
    page_size_in_bytes = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])

    # Calculate the page-aligned address
    page_aligned_address = (virtual_address // page_size_in_bytes) * page_size_in_bytes
    return page_aligned_address

def get_page_size_in_gb():
    page_size_in_bytes = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])
    page_size_in_gb = page_size_in_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    return page_size_in_gb

def get_tensor_pages(tensor):
    tensor_size = get_tensor_size_in_gb(tensor)
    page_size = get_page_size_in_gb()
    pages_in_tensor = tensor_size / page_size
    return int(pages_in_tensor)

pack_id = 1
unpack_id = 1

last_objects = {}
dtype_id = {}

total_size = 0

def pack_hook(x):
    if x is None:
        return x
    global total_size
    
    total_size += int(get_tensor_size_in_gb(x))
    # if compression:
    #     if get_tensor_size_in_gb(x) > :
    #         if x.dtype == torch.float32:
    #             x_new = x.float()
    #             x_new = torch.quantize_per_tensor(x_new, scale=0.1, zero_point=0, dtype=torch.qint8)
    #             x = x_new
    #             print(f"Data has been compressed!")

    global pack_id
    with open("log_files/obj_dump.log", 'a') as file:
        if get_tensor_size_in_gb(x) > 0.0001:
            file.write(f"Object: FWD_Output_Cache{pack_id}: {get_page_aligned_address(x.data_ptr())}, {get_tensor_pages(x)} \n")
    if get_tensor_size_in_gb(x) > 0.0001:
        # print("------------------------------------------------")
        # print("Packing")
        # print(f"Address of tensor = {get_page_aligned_address(x.data_ptr())}\n")
        # print_tensor_size_in_gb(x)
        # print("------------------------------------------------")
        pack_id += 1
    return x

def unpack_hook(x):
    if x is None:
        return x
    global unpack_id 
    # if compression:
    #     if x.dtype == torch.qint8:
    #         x = x.dequantize()

    with open("log_files/obj_dump.log", 'a') as file:
        if get_tensor_size_in_gb(x) > 0.0001:
            file.write(f"Object: BWD_Input_Cache{unpack_id}: {get_page_aligned_address(x.data_ptr())}, {get_tensor_pages(x)} \n")
    if get_tensor_size_in_gb(x) > 0.0001:
        # print("------------------------------------------------")
        # print("Unpacking")
        # print(f"Address of tensor = {get_page_aligned_address(x.data_ptr())}\n")
        # print_tensor_size_in_gb(x)
        # print("------------------------------------------------")
        unpack_id += 1
    return x

# ======= CNN Definition =======
class MyCNN(nn.Module):
    def __init__(self, num_layers, num_classes):
        super(MyCNN, self).__init__()
        if num_layers == 18:
            self.model = torchvision.models.resnet18(num_classes=num_classes)
        elif num_layers == 34:
            self.model = torchvision.models.resnet34(num_classes=num_classes)
        elif num_layers == 50:
            self.model = torchvision.models.resnet50(num_classes=num_classes)
        elif num_layers == 152:
            self.model = torchvision.models.resnet152(num_classes=num_classes)
        else:
            raise ValueError("Unsupported ResNet version.")

        self.layers = 0
        for name, module in self.model.named_children():
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_backward_hook(backward_hook)
            print(f"name: {name}, module: {module}")
            self.layers += 1

        print(f"Number of layers = {self.layers}")

    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            x = self.model(x)
            return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MyCNN(args.resnet_layers, num_classes).to(device)
layers = net.layers
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# # ======= Training Loop =======
# for epoch in range(args.epochs):
#     net.train()
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data[0].to(device), data[1].to(device)

#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

def train():
    with open("log_files/obj_dump.log", 'a') as file:
        file.write(f"Layers : {layers}\n")

    epochs_time = []

    # Load the entire dataset once as full batch
    inputs, labels = next(iter(torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)))

    inputs, labels = inputs.to(device), labels.to(device)

    for epoch in range(args.epochs):
        global total_size
        total_size = 0
        with open("log_files/obj_dump.log", 'a') as file:
            file.write(f"Epoch : {epoch + 1}\n")

        net.train()
        optimizer.zero_grad()

        t1 = time.time()

        add_marker("FWD_START", "red")
        outputs = net(inputs)
        add_marker("FWD_END", "yellow")

        t3 = time.time()

        loss = criterion(outputs, labels)

        add_marker("BWD_START", "blue")
        loss.backward()
        add_marker("BWD_END", "blue")

        t2 = time.time()
        epochs_time.append(t2-t1)

        optimizer.step()

        print(f"Epoch {epoch + 1} finished. Loss: {loss.item():.4f}")
        print(f"Runtime: {t2-t1:.2f}s | FWD time = {t3-t1:.2f}s | BWD time = {t2-t3:.2f}s")
        print(f"Total size of saved tensors in epoch {epoch + 1} = {total_size} GB")

    avg_epoch_time = sum(epochs_time) / len(epochs_time) if epochs_time else 0
    print(f"Average Epoch Runtime = {avg_epoch_time:.2f}s")


if __name__ == '__main__':
    train()
    print("Training completed.")
