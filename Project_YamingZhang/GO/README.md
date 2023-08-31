# GO
First load the module and compile the code:
```
srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
module load go
```

Run the code:
```
go run main.go
```