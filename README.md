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

I'm having success with this SFT configuration using a dataset of 7500 instruction, output pairs that are in total 870k tokens. If you open the sftune.py script you will see a variable at the top called `num_train_epochs` that is a multiplier so I'm essentially doubling this token count per epoch.

### Installation note

I want pip install to work from the requirements.txt I have included here but sadly that rarely works so I'd ignore that detail here and start with [unsloth](https://github.com/unslothai/unsloth) to be sure you have a solid installation.
