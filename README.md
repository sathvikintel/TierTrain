# TierTrain

## Build steps

### Clone the repo

```
git clone https://github.com/sathvikintel/TierTrain.git
```

### Change dir

```
cd TierTrain
```

### Compile TierTrain daemon
```
gcc -o tier_train_daemon tier_train_daemon.c -lpthread -lrt -lnuma
```


### Run workload

```
./train_resnet_tt.sh <num_layers (18, 34, 50, 152)> <num_epochs> <dataset (cifar10/cifar100)> <run tiertrain? (0/1)>
```

### Notes

- To change fast and slow memory nodes, edit below lines in `tier_train_daemon.c`
```
#define FAST_MEM_NODE 0
#define SLOW_MEM_NODE 3
```

- Full batch mode training chosen to blow up footprint


