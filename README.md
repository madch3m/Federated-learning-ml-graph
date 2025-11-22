# Federated Learning ML graph

> An implementation of the paper by McMahan, Moore, Ramage, et.al. on "Communication-Efficient Learning of Deep Networks from Decentralized Data"[^1].

[^1]: McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.


***
## How to run:

### Using Docker (Recommended):
##### On MacOS/Linux:
```bash
# Run with default CNN model
chmod +x setup.sh && ./setup.sh --docker

# Run with CIFAR10 CNN model
./setup.sh --docker --model cifar10_cnn

# Run with ResNet model
./setup.sh --docker --model resnet

# Or use short flags
./setup.sh -d -m cifar10_cnn
```

##### On Windows PowerShell:
```powershell
# Run with default CNN model
.\setup.ps1 --docker

# Run with specific model
.\setup.ps1 --docker --model cifar10_cnn
```

### Using Local Python Environment:
##### On MacOS/Linux:
```bash
# Setup and run with default CNN model
chmod +x setup.sh && ./setup.sh --run

# Run with specific model
./setup.sh --run --model cifar10_cnn

# Just setup without running
./setup.sh
source .venv/bin/activate
```

##### On Windows PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

### Available Models:
- `cnn` - Simple CNN for MNIST (default)
- `cifar10_cnn` - CNN for CIFAR10 dataset
- `resnet` - ResNet architecture

### Help:
```bash
./setup.sh --help
```
***
Since all packages can eat up a lot of space...
## How to delete everything after:
##### On MacOS/Linux:
```
rm -rf .venv
rm -rf ./data
pip cache purge
rm -rf ~/.cache/torch
rm -rf ~/.cache/matplotlib
```

##### On Windows PowerShell:
```
Remove-Item .venv -Recurse -Force
Remove-Item .\data -Recurse -Force
pip cache purge
Remove-Item "$env:USERPROFILE\.cache\torch" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\matplotlib" -Recurse -Force -ErrorAction SilentlyContinue
```