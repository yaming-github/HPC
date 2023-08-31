# JULIA
First load the module and compile the code:
```
srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
module load julia
```

Run the code:
```
JULIA_NUM_THREADS=16 julia main.jl
```