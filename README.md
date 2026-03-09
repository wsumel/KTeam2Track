
# KTeam2Track

Official repository for **KTeam2Track**.


## Reproduction

1. Clone the repository and configure the environment by following the project instructions.
2. Download the pretrained weights and place them in the specified directory.
3. Modify the paths in `test.sh` according to your local setup, including the test dataset path (`DIR`), output directory (`OUTPUT_DIR_BASE`), and checkpoint path (`ck_pth`).
4. Run the test script:

## Install the environment
```
conda create -n kteamtrack python=3.11
conda activate kteamtrack
bash install.sh
```

* Add the project path to environment variables





## Weights


- **Pretrained Weights**: [https://drive.google.com/file/d/183W6xX882s5NXMN_C0COliBnNrF_GEfe/view?usp=drive_link](https://drive.google.com/file/d/183W6xX882s5NXMN_C0COliBnNrF_GEfe/view?usp=drive_link)



```bash
bash test.sh
```