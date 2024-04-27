Mistral 7B chat fine tuning

### SFT with unsloth

```
git clone git@github.com:toranb/sloth.git
cd sloth
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
## add data.json with instruction, output pairs for supervised fine tune
python3 sftune.py
```

### Merge from checkpoint (optional)

This cmd will merge a given checkpoint, creating a new model directory

```
rm -rf model
python3 zmerge.py --peft /home/toranb/sloth/workspace/checkpoint-2600
```

### DPO alignment (optional)

```
mkdir fin
export DPO=/home/toranb/sloth/model
export JSON=/home/toranb/sloth/dpo.json
export OUTPUTDIR=/home/toranb/sloth/fin
## add dpo.json with prompt, chosen, rejected
python3 dpo.py --base $DPO --out $OUTPUTDIR --json $JSON
```

### Dataset note

I'm having success with this SFT configuration using a dataset of 21k instruction, output pairs that are in total 3MIL tokens. This 21k dataset is a combination of 10k from a subset of [airoboros](https://huggingface.co/datasets/jondurbin/airoboros-3.1) and 11k from a proprietary dataset.

### Installation note

I want pip install to work from the requirements.txt I have included here but sadly that rarely works so I'd ignore that detail here and start with [unsloth](https://github.com/unslothai/unsloth) to be sure you have a solid installation.

As of April 2024, flash-attn has a problem so I'm using 2.5.6 to workaround the installer like so
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install scipy
pip install trl
pip install xformers
pip install wandb
pip install packaging ninja einops
pip install trl peft accelerate bitsandbytes
pip install flash-attn==2.5.6
pip install "unsloth[cu118-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade bitsandbytes
```
