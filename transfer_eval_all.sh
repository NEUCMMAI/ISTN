#!/bin/bash
###
 # @Author: Kang Yang
 # @Date: 2023-07-28 20:54:03
 # @LastEditors: Kang Yang
 # @LastEditTime: 2023-07-29 19:35:37
 # @FilePath: /project-b/transfer_eval_all.sh
 # @Description: 
 # Copyright (c) 2023 by Kang Yang, All Rights Reserved. 
### 

# modelRawResult=`cat /workspace/project-b/log/B/B-None-Jul-28-2023_15-21-07-871973.log` 
# result=`echo "$abb" | grep "test result"`

# modelList=(B DIFSR IGCMC IGMC LightSANs RetaGNN SINE) #no S3Rec
model=IGCMC #no S3Rec
datasetList=(foursquare_NYC foursquare_TKY ml-1m ml-100k tmall-buy yelp2018 yelp2022)

for i in ${datasetList[@]}
    do
        for j in ${datasetList[@]}
            do
                if [ $i == $j ]
                then
                    continue
                fi
                python run.py --model=$model --source_dataset=$i --target_dataset=$j --source_weight_path="checkpoints/$model-$i.pth" --only_eval_target=True --only_output_test=True
                # modelRawResult=`python run.py --model=$model --source_dataset=$i --target_dataset=$j --source_weight_path="checkpoints/$model-$i.pth" --only_eval_target=True`
                # result=`echo "$modelRawResult" | grep "test result"`
                # echo -e "$model-$i->$j: \n $result \n"  >> log.out
            done
    done