# Q1
We have a sbatch script in the dir.<br>
e.g.<br>
First load the module and compile the code:<br>
NOTE: please do not change the name of output bin file
```
module load openmpi
mpicc Q1.c -o Q1
```

Run the MPI task with Q1.script:
```
sbatch Q1.script
```