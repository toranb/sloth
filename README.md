Mistral 7B chat fine tuning

### SFT with unsloth

```
git clone git@github.com:toranb/sloth.git
cd sloth
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
## add data.json with instruction, output pairs for supervised fine tune
python3.11 sftune.py
```

### Merge from checkpoint (optional)

This cmd will merge a given checkpoint, creating a new model directory

```
rm -rf model
python3.11 zmerge.py --peft /home/toranb/sloth/workspace/checkpoint-2600
```

### DPO alignment (optional)

```
mkdir fin
export DPO=/home/toranb/sloth/model
export JSON=/home/toranb/sloth/dpo.json
export OUTPUTDIR=/home/toranb/sloth/fin
## add dpo.json with prompt, chosen, rejected
python3.11 dpo.py --base $DPO --out $OUTPUTDIR --json $JSON
```

### Dataset note

I'm having success with this SFT configuration using a dataset of 21k instruction, output pairs that are in total 3MIL tokens. This 21k dataset is a combination of 10k from a subset of [airoboros](https://huggingface.co/datasets/jondurbin/airoboros-3.1) and 11k from a proprietary dataset.

### Installation note

I want pip install to work from the requirements.txt I have included here but sadly that rarely works so I'd ignore that detail and start with [unsloth](https://github.com/unslothai/unsloth) to be sure you have a solid installation.

March 2025: I've had success installing unsloth with uv using these steps with CUDA 12.6 & torch 2.5.1
```
uv python install 3.11
uv venv
source .venv/bin/activate
uv pip install "unsloth[cu126-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
uv pip install torch==2.5.1 xformers ninja setuptools wheel sentencepiece
uv pip install --no-deps "trl<0.9.0" numpy pytz pandas peft accelerate bitsandbytes
uv pip install datasets transformers
uv pip install rich click pydantic unsloth_zoo
uv pip install flash-attn --no-build-isolation
```
