# DebiasedDTA - Reproduce

⚠**Important Note**⚠ This repository contains the **exact** codes that produced the results in the manuscript and is open-sourced only for scientific openness. If you are more interested in using the debiasing methods than reproducing the manuscript, please refer to [this link](https://github.com/boun-tabi/debiaseddta) that contains an easy-to-use python library. If you would like to replicate the results, please first install the dependencies with `pip install requirements.txt` and then refer to the example commands below. In the cases you encountered `NameError` while importing code from `src` module, add the path of the directory to your `PYTHONPATH`. 

 If you use this repository in your research, please cite...

```
xxxx
```

---

### Hyper-Parameter Search Examples

```run_hp_search.py``` is used to tune the parameters of the strong leaners (DeepDTA, BPE-DTA, and LM-DTA). Here are some example commands you can use to start hyper-parameter search. 

- Use the following command in order to tune a DeepDTA model on BDB dataset. Set the second parameter as `kiba` to change the dataset.

	```python src/run_hp_search.py deepdta bdb```

- In order to tune a BPE-DTA model on BDB dataset and also search for the optimum vocabulary size, use the command below. The command automatically changes the chemical and protein vocabulary sizes from 8K to 32K, experimenting with every combination.
  
    ```python src/run_hp_search.py bpedta bdb```
    
- Last, the following command will tune the hyper-parameters of LM-DTA on KIBA dataset.
  
  ```python src/run_hp_search.py lmdta kiba```
### Debiasing Examples

```debias.py``` is the main file used to train a strong learner with debiasing. Here are some usage examples. 

- The following command debiases BPE-DTA (vocab sizes of 8K and 32K) using BoWDTA. The training dataset is selected as BDB and weight adaptation strategy is set to BD by `lin_decrease` parameter.

  ```python src/debias.py  --model_name bpedta --weak_learner_name bowdta --chem_vocab_size 8000 --prot_vocab_size 32000 --dataset bdb --n_bootstrapping 10 --decay_type lin_decrease```

 - Here is another example to debiases LM-DTA with IDDTA on KIBA dataset. The weight adaptation strategy is set to BG this time and the affinity score scaling is turned on. Pre-computed ligand and protein embedding paths are provided for faster model creation.

   ```python src/debias.py  --model_name lmdta --weak_learner_name iddta --dataset kiba --n_bootstrapping 10 --decay_type lin_increase --scale --lm_ligand_path ligand_embed.json --lm_protein_path protein_embed.json```
