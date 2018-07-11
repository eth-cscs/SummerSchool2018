Setting up the environment for building the miniapp:

```
# to build

module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
make -j4

# to run

srun -n srun ./main 128 128 128 0.01

# to plot

module load PyExtensions/2.7-CrayGNU-17.08
python2 ../plotting.py
```

If you have an interactive session, you can uncomment the line `srun ./unit_tests`, which will run the unit tests every time you compile.
