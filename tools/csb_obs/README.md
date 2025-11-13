# csb obs下载脚本

### 特性
 
 - 增量下载（当前根据文件大小进行判断是否需要重新下载）
 - 多进程
 - 文件类型过滤（多个文件类型之间使用英文逗号隔开）
 
 
### 使用方法
 
```
python yellow_folder_downloader.py \
    --app_token=your_app_token \
    --vendor=HEC \
    --region=cn-south-1 \
    --bucket_name=bucket-7769-huanan \
    --path=autoML/outputs/some_path/ \
    --objects_storage_path=. \
    --processes=88  \ # 多进程下载
    --exclude=.pt,.ckpt,ascend_log # 过滤这些后缀或路径包含这些关键词不下载