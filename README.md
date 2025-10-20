# Federated Learning ML graph

> An implementation of the paper by McMahan, Moore, Ramage, et.al. on "Communication-Efficient Learning of Deep Networks from Decentralized Data"[^1].

[^1]: McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.


***
## How to run:
##### On MacOS/Linux:
```
chmod +x setup.sh && ./setup.sh --run
source .venv/bin/activate
```

##### On Windows PowerShell:
```
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
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