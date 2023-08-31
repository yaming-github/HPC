# Q4
First load the module and compile the code:
```
module add hpc_sdk
module load cuda/11.7
nvcc GPUvecAdd.cu -o Q4CUDA
pgcc -acc -Minfo=accel Q4.c -o Q4ACC
```

Run the OpenACC code:
```
Allocate a GPU node first and run the code.
./Q4CUDA
./Q4ACC
```