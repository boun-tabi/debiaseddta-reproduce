# DebiasedDTA - Reproduce

⚠**Important Note**⚠ This repository is the **exact** codebase that produced the results in the [manuscript](https://arxiv.org/abs/2107.05556) and is open-sourced for scientific openness. If you would like to replicate the results, please first clone repository and then install the dependencies with `pip install -r requirements.txt` . You can use the commands below to use/replicate the methods in the paper. We will soon release a Python library to further ease the use of these models. ⏰

 If you use any part of this repository in your research, please cite:

```
@misc{özçelik2021debiaseddta,
      title={DebiasedDTA: Model Debiasing to Boost Drug -- Target Affinity Prediction}, 
      author={Rıza Özçelik and Alperen Bağ and Berk Atıl and Arzucan Özgür and Elif Özkırımlı},
      year={2021},
      eprint={2107.05556},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```

---

### Debiasing Examples

```debias.py``` is the main file used to train a strong learner with debiasing. Here are some usage examples. 

- The following command debiases BPE-DTA (vocab sizes of 8K and 32K) using BoWDTA. The training dataset is selected as BDB and weight adaptation strategy is set to BD by `lin_decrease` parameter.

  ```python src/debias.py  --model_name bpedta --weak_learner_name bowdta --chem_vocab_size 8000 --prot_vocab_size 32000 --dataset bdb --n_bootstrapping 10 --decay_type lin_decrease```

 - Here is another example to debiases LM-DTA with IDDTA on KIBA dataset. The weight adaptation strategy is set to BG this time and the affinity score scaling is turned on. Pre-computed ligand and protein embedding paths are provided for faster model creation.

   ```python src/debias.py  --model_name lmdta --weak_learner_name iddta --dataset kiba --n_bootstrapping 10 --decay_type lin_increase --scale --lm_ligand_path ligand_embed.json --lm_protein_path protein_embed.json```


### Hyper-Parameter Search Examples

```run_hp_search.py``` is used to tune the parameters of the strong leaners (DeepDTA, BPE-DTA, and LM-DTA). Here are some example commands you can use to start hyper-parameter search. 

- Use the following command in order to tune a DeepDTA model on BDB dataset. Set the second parameter as `kiba` to change the dataset.

	```python src/run_hp_search.py deepdta bdb```

- In order to tune a BPE-DTA model on BDB dataset and also search for the optimum vocabulary size, use the command below. The command automatically changes the chemical and protein vocabulary sizes from 8K to 32K, experimenting with every combination.
  
    ```python src/run_hp_search.py bpedta bdb```
    
- Last, the following command will tune the hyper-parameters of LM-DTA on KIBA dataset.
  
  ```python src/run_hp_search.py lmdta kiba```
