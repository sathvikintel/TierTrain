#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <no_of_layers (18, 34, 50, 152)> <no_of_epochs> <dataset (cifar10, cifar100)> <tier_train (0/1)>"
    exit 1
fi

num_layers="$1"
num_epochs="$2"
dataset="$3"
tier_train="$4"

torch_tt_comm_file="log_files/obj_dump.log"
migration_log_1="log_files/migration_stats_1.log"
migration_log_2="log_files/migration_stats_2.log"

> $torch_tt_comm_file
> $migration_log_1
> $migration_log_2

initial_time=$(date +%s)
output_file="log_files/utc_start_time.txt"
> $output_file
echo "$initial_time" > "$output_file"

# Run resent training workload
numactl  --cpunodebind=0 --interleave=0  python train_resnet.py --dataset $dataset --resnet_layers $num_layers --epochs $num_epochs &
train_resnet_pid=$!

echo "Workload PID: $train_resnet_pid"

if [ $tier_train = 1 ] ; then
    numactl --cpunodebind=0 ./tier_train_daemon $torch_tt_comm_file $migration_log_1 $migration_log_2 $train_resnet_pid &
fi

wait 

echo "train_resnet_tt.sh: Run over"

