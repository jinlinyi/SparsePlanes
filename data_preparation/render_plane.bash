#!/bin/bash
max=5
for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    let num=18
    let base=$i*$num
    CUDA_VISIBLE_DEVICES=$i nice -n 10 python render_plane.py --base=$base --num=$num --id=$i & pid=$!
    PID_LIST+=" $pid";
} done
trap "kill $PID_LIST" SIGINT
echo "Parallel processes have started";
wait $PID_LIST
echo
echo "All processes have completed";