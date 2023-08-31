# C
First load the module and compile the code:
```
module load openblas/0.3.6
module load cuda
gcc -pthread -fopenmp -lopenblas -lm main.c -o main
nvcc main.cu -o maincuda
```

Run the code:
```
srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
./main
./maincuda
```