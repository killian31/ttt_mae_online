# Test Time Training with Masked Autoencoders: Further Experiments
⚠️Work in development⚠️

This repo aims at reproducing
[ttt-mae](https://yossigandelsman.github.io/ttt_mae/index.html) results and
experiment further.

## Install

1. Create a Python virtual environment

```bash
pyenv virtualenv 3.12.2 ttt-online
pyenv activate ttt-online
```

2. Install the dependencies

```bash
pip install poetry
poetry install
```

## Run

```bash
poetry run python main.py --help
```

## State of Development

- [x] Run test time training
- [ ] Run online test time training
- [ ] Characterize failure cases
- [ ] Impact of the number of steps
