# ISSAI

---

## Getting Started

Install APT packages:
```sh
apt install python3 python3-pip python3-venv python3-tk
# apt install python3.8
```

Create and use environment:
```sh
python3 -m venv .venv
# -m - module-name, finds sys.path and runs corresponding .py file
source .venv/bin/activate
```

Install python libraries:
```sh
pip install torch torchvision torchaudio
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
```

To recreate environment:
```sh
pip freeze > requirements.txt
```

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
