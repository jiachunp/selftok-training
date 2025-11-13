# SelfTok Developer Guide
The guide is intended for developers of SelfTok tokenizer project, where "developers" refer to anyone that intend to modify the codebase, and "SelfTok tokenizer" refers to the pre-training stage of SelfTok tokenizer (*e.g.*, training a GPT on SelfTok tokens is out of scope).

The guide assumes that the user has basic knowledge of operating the ROMA platform, which manages the computing resources and cloud storage. You should know how to apply for a debug node, how to connect to the debug node (*e.g.*, using VSCode) and how to access data stored on the cloud (*i.e.*, S3). If you are uncertain about these, please refer to xxx (guide by dongze) first. The guide also requires knowledge of common developer tools, *e.g.*, VSCode as the IDE and Git for version control. If you are unfamilar with those tools, please check out online guides.

The guide is divided into two parts. The first part introduces the development workflow, from setting up the server, running debug and training, to viewing results and merging with codebase. The second part

## 1. Development Workflow
A general workflow for modifying codebase consists of the following steps:

### 1.1 Setup
If you do not have a debug node, apply one first. See the illustrative guide in xxx. After acquiring the node, perform the following actions:

#### 1.1.1 Debug Node with NPUs
1. Mount <obs://bucket-6824-huanan/outputs/selftok_enc_tb/> to **/data/tb/** on ROMA DevContainer. This is for viewing tensorboard logs from existing experiments.
2. Pull the codebase by xxxx.
3. Run the following command to download required models.
```shell
    cd /home/ma-user/work/SelfTokEn
    python ./scripts/setup.py
```
4. Install additional required packages by running the following in a terminal:
```shell
    conda activate python-3.9.10
    pip install tensorboard
    conda activate PyTorch-2.1.0
    pip install lpips
```
5. (Optional) To conveniently view reconstructed images, create a symbolic link by:
```shell
    cd /home/ma-user/work/SelfTokEn
    ln -s /data/tb/ ./logs
```

#### 1.1.2 Debug Node with GPUs
1. Mount <obs://bucket-9122-wulan/outputs/selftok_enc_tb/> to **/data/tb/** on ROMA DevContainer. This is for viewing tensorboard logs from existing experiments.
2. Pull the codebase by xxxx.
3. Run the following command to download required models.
```shell
    cd /home/ma-user/work/SelfTokEn
    python ./scripts/setup_gpu.py
```

### 1.2 Working on Your Git Branch
WIP (pending code repo setup).

### 1.3 Single-Card Debug
Debug on a single card is the best way to find code errors fast. It also supports debugging with breakpoints. Note that it cannot track down problems about multi-NPU or multi-node training.

The debug configurations are stored in **.vscode/launch.json**. Two default configurations for GPU and NPU are provided. Use the corresponding configuration based on your device type. Change **yml_path=XXXX.yml** to the desired config file.

To start debugging, first press Ctrl+Shift+P and select "Python: Select Interpreter".
For NPU, select "PyTorch-2.1.0" from the drop-down list.
For GPU, select "Enter interpreter path" and type *"/home/ma-user/anaconda3/envs/tokenizer/bin/python"*.
Click the "Run and Debug" tab on the left, select the debug configuration, and click run.


### 1.4 Multi-Card Debug

Run the following command by replacing "xx" with the correct value. Note that id field is compulsory.
```shell
# NPU
python scripts/debug_selftok_enc.py --id=xx --config=encoder/v1.1 --num_gpus=xx
# GPU
python scripts/debug_selftok_enc_gpu.py --id=xx --config=encoder/v1.1 --num_gpus=xx
```
After killing a running debug program (*e.g.*, by Ctrl+C), it is common that some processes are not stopped automatically. In such case, it is necessary to do a manual clean-up to prevent failure of future debug programs. To do so, run the following commands:
```shell
bash scripts/pkill_node
nvidia-smi  # for GPUs. Run "npu-smi info" instead for NPUs.
kill -9 XX
...
```
Run the kill commands on all running process names found in nvidia-smi (or npu-smi info).




### 1.5 Launch Training


### 1.6 View Logs

1. Make sure <obs://BUCKET-NAME/outputs/selftok_enc_tb/> is mounted at /data/tb/ on ROMA DevContainer.
2. Run the following command on a DevContainer terminal
```shell
    # Debug node with NPUs
    conda activate python-3.9.10
    pip install tensorboard
    tensorboard --logdir=/data/tb --port=6006
    # Debug node with GPUs
    conda activate PyTorch-1.10.2
    tensorboard --logdir=/data/tb --port=6006
```
3. Replace XXXXX with notebook ID (find in ssh config file), and run following command on PC
```shell
    ssh -L 16006:127.0.0.1:6006 ModelArts-Note-XXXXX
```
4. Open a browser and type in http://127.0.0.1:6006/

### Launch Validation


### Merging Changes with Codebase


## Codebase

### File Structure
gfgafsdfadsf

    configs
    ├── mimo
    │   ├── selftok
    │   │   ├── base_NAME1.yml              # base setting 1
    │   │   ├── base_NAME2.yml              # base setting 2
    │   │   ├── NAME1_v1.0.yml              # v1.0 modifier for base setting 1
    │   │   └── ...
    │   └── ...
    └── ...
    mimogpt
    ├── datasets
    │   ├── selftok_dataset.py              # latent/image dataset & dataloaders
    │   └── ...
    ├── engine
    │   ├── utils
    │   │   ├── selftok_hook.py             # opt, sch, threshold, ema, save, log
    │   │   ├── selftok_validation.py       # validation step
    │   │   └── ...
    │   ├── trainer_selftok_enc.py          # init, train step, prepare logs
    │   └── ...
    ├── models
    │   ├── selftok
    │   │   ├── diffusion
    │   │   │   ├── gaussian_diffusion.py   # diffusion process
    │   │   │   ├── respace.py              # diffusion with time-step subset
    │   │   │   ├── timestep_sampler.py     # ways of sampling time-steps
    │   │   │   └── ...
    │   │   ├── image_tokenizer.py          # enc-dec model
    │   │   ├── models_ours.py              # enc, dec and their variants
    │   │   ├── models.py                   # base models
    │   │   └── diti_utils.py               # 1...K <=> 1...T
    │   └── ...    
    ├── tokenizer
    │   ├── selftok
    │   │   ├── vector_quantize_pytorch.py  # default VQ
    │   │   ├── OTHER_QUANTIZER1.py         # other VQs
    │   │   └── ...
    │   └── ...
    └── ...
    scripts
    ├── setup.py                            # setup debug node
    ├── debug_selftok_enc.py                # run multi-npu debug
    ├── val_selftok_enc.py                  # val selftok ckpt
    ├── draw_log.py                         # draw DM loss curve (depreciating)
    ├── upload_code.py                      # upload code to S3 (depreciating)
    └── pkill_node.sh                       # kill train processes
    train_net.py
    train_cloud_gpu.py
    train_cloud_ascend.py
    train_cloud_ascend_debug.py

Note:
1. Inside MIMO
2. Irrelevant folders are ignored