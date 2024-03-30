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


### DPO alignment (optional)

```
mkdir fin
export DPO=/home/toranb/sloth/model
export JSON=/home/toranb/sloth/dpo.json
export OUTPUTDIR=/home/toranb/sloth/fin
## add dpo.json with prompt, chosen, rejected
python3 dpo.py --base $DPO --out $OUTPUTDIR --json $JSON
```

### Installation note

I want pip install to work from the requirements.txt I have included here but sadly that rarely works so I'd ignore that detail here and start with [unsloth](https://github.com/unslothai/unsloth) to be sure you have a solid installation.
