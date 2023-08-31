# Q2
We have 4 scripts in the dir.<br>
e.g.<br>
First we need to load the openmpi and compile the code:<br>
NOTE: please do not change the output bin file name
```
module load openmpi
mpic++ --std=c++11 Q2.cpp -o Q2
```

Run the MPI task with 100 bins and 2 nodes and 4 nodes:
```
sbatch Q2_100_2.script
sbatch Q2_100_4.script
```

Run the MPI task with 20 bins and 2 nodes and 4 nodes:
```
sbatch Q2_20_2.script
sbatch Q2_20_4.script
```