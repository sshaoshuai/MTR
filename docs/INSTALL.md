# Installation
The installation of MTR is super easy.

If you would like to install it to an existing PyTorch environment, just jump to **step 3**. 

**Step 1:** Create an python environment for MTR
```shell
conda create --name mtr python=3.8 -y
conda activate mtr 
```

**Step 2:** Install the required packages
```shell
pip install -r requirements.txt
```

**Step 3:** Clone the codebase, and compile this codebase: 
```python
# Clone MTR codebase
git clone https://github.com/sshaoshuai/MTR.git

# Compile the customized CUDA codes in MTR codebase
python setup.py develop
```

Then you are all set.
