Thrust mini-app:  Setting up the environment

```
# to build

module load daint-gpu
module load craype-accel-nvidia60
make -j4

# to run

salloc -N 1 -C gpu --partition=debug
srun -n 1 ./main 128 128 128 0.01

# to plot

module load PyExtensions/2.7-CrayGNU-17.08
python2 ../plotting.py
```

If you have an interactive session, you can uncomment the line `srun ./unit_tests`, which will run the unit tests every time you compile.
