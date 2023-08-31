# Q3
First load the module and compile the code:
```
module load cuda
nvcc Q3.cu -o Q3
```

Run the cuda matmul:
```
sbatch Q3.script
```