PROCESS_NUM=8
 
# eg: /data/x00520376/code/video/workspace/acclerate/val2014.zip
local_folder_path=$1
 
# eg: data/coco2014/
bucket_path=$2
 
for ((process_idx=0;process_idx<$PROCESS_NUM;process_idx+=1));
do   
    python s3_uploader.py \
        --local_folder_absolute_path=$local_folder_path \
        --app_token=bd43698f-f400-4f2a-8a09-a40238cb6607 \
        --vendor=HEC --region=cn-north-4 \
        --bucket_name=bucket-vedata02-bj4 \
        --bucket_path=$bucket_path \
        --multi_process=$PROCESS_NUM \
        --process_idx=$process_idx \
        --incremental=1 \
        --thread_num=24 --show_speed &
done