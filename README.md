# Active Generalized Category Discovery with Diverse LLM Feedback

## Setup
```bash
conda create -n gcdllms python=3.8 -y
conda activate gcdllms

# install pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# install dependency
pip install -r requirements.txt
pip install faiss-gpu==1.7.2
pip install boto3 # if you need to use Claude and LLama provided by Bedrock
```

## Running
First, add you OpenAI API key in line 61 of the 'run_GCDLLMs.sh' file.

Pre-training, training and testing our model through the bash script:
```bash
sh run_GCDLLMs.sh
```
You can also add or change parameters in run_GCDLLMs.sh (More parameters are listed in init_parameter.py)


## Bugs or Questions

If you have any questions related to the code or the project, feel free to email Henry Peng Zou ([pzou3@uic.edu](pzou3@uic.edu), [penzou@amazon.com](penzou@amazon.com)). If you encounter any problems when using the code, or want to report a bug, please also feel free to reach out to us. Please try to specify the problem with details so we can help you better and quicker!

## Acknowledgement
This repo borrows some data and codes from [Loop](https://github.com/Lackel/LOOP) and [JointMatch](https://github.com/HenryPengZou/JointMatch). We appreciate their great works!