#!/bin/bash
max=45
for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    let num=2
    let base=$i*$num
    nice -n 10 python convert_mp3d_ply.py --base=$base --num=$num --id=$i & pid=$!
    PID_LIST+=" $pid";
} done
trap "kill $PID_LIST" SIGINT
echo "Parallel processes have started";
wait $PID_LIST
echo
echo "All processes have completed";