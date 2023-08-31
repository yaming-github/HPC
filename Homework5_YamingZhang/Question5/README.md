# Q5
First load the module and compile the code:
```
module add hpc_sdk
module load cuda/11.7
pgcc -acc -Minfo=accel Q5.c -o Q5
```

Run the OpenACC code:
```
Allocate a GPU node first and run the code.
./Q5 1024
```