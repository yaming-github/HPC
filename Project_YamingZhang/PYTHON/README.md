# PYTHON
First load the module and compile the code:
```
srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
module load anaconda3/2022.01
module load cuda/11.1
source activate pytorch_env_training
```

Run the code:
```
python main.py
```