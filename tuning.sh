#!/bin/bash

for i in {0..10}
do
    for j in {1..16}
    do
        echo $1"."$i"."$j >> $1".log"
        python main.py -d $1 -m $i"."$j | tail -1  >> $1".log"
    done
    
    echo $1"."$i >> $1".log"
    python main.py -d $1 -m $i | tail -1  >> $1".log"
done
