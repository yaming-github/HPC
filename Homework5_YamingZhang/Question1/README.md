# Q1
First load the module and compile the code:
```
nvcc --std=c++11 Q1.cu -o Q1
g++ --std=c++11 -fopenmp Q1.cpp -o Q11
```

Run the OpenACC code:
```
Allocate a GPU node first and run the code.
./Q1
./Q11
```