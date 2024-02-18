Mistral 7B chat fine tuning

### Pull down Mistral 7B instruct

```
git clone --depth=1 https://github.com/Lightning-AI/lit-gpt lit
cd lit
python3 -m venv env
source env/bin/activate
pip install -r requirements-all.txt
python3 scripts/download.py --repo_id mistralai/Mistral-7B-Instruct-v0.2
python3 scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.2
```


### SFT (new terminal)

```
cd
mkdir workspace
cd workspace
mkdir wat
mkdir dpo
```

```
cd
git clone git@github.com:toranb/sloth.git
cd sloth
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
## add data.json with instruction, output pairs for supervised fine tune
export MIS=/home/toranb/lit/checkpoints/mistralai/Mistral-7B-Instruct-v0.2
python3 sftune.py
```


### Merge adapter to a new model

```
python3 zmerge.py --base $MIS --peft /home/toranb/workspace/checkpoint-??? --out /home/toranb/workspace/wat
```

### DPO alignment (optional)

```
mkdir fin
export DPO=/home/toranb/workspace/wat
export JSON=/home/toranb/sloth/dpo.json
export OUTPUTDIR=/home/toranb/sloth/fin
## add dpo.json with prompt, chosen, rejected
python3 dpo.py --base $DPO --out $OUTPUTDIR --json $JSON
```


### Merge adapter to a new model

```
python3 zmerge.py --base $DPO --peft /home/toranb/sloth/fin/checkpoint-??? --out /home/toranb/workspace/dpo
```
