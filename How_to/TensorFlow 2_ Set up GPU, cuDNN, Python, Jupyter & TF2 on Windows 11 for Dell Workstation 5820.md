![Untitled](https://github.com/perceptionlab1/AISilicosis/assets/141559457/f1a72651-fd10-4fe0-bfc6-dd07c7825886)
![Uploading Untitled 17.png…]()
![Uploading Untitled 16.png…]()
![Uploading Untitled 15.png…]()
![Uploading Untitled 14.png…]()
![Uploading Untitled 13.png…]()
![Uploading Untitled 12.png…]()
![Uploading Untitled 11.png…]()
![Uploading Untitled 10.png…]()
![Uploading Untitled 9.png…]()
![Uploading Untitled 8.png…]()
![Untitled 7](https://github.com/perceptionlab1/AISilicosis/assets/141559457/be97d05c-d98a-417a-b537-c51607ce3573)
![Untitled 6](https://github.com/perceptionlab1/AISilicosis/assets/141559457/8d034c25-dd6a-4e24-bc64-5a98aae8d582)
![Uploading Untitled 5.png…]()
![Uploading Untitled 4.png…]()
![Uploading Untitled 3.png…]()
![Uploading Untitled 2.png…]()
![Uploading Untitled 1.png…]()
![Uploading Untitled 22.png…]()
![Uploading Untitled 21.png…]()
![Uploading Untitled 20.png…]()
![Uploading Untitled 19.png…]()
![Uploading Untitled 18.png…]()
![Uploading tf.png…]()

# Set up GPU, cuDNN, Python, Jupyter & TF2 on Windows 11 for Dell Workstation 5820

## Check if you have all the components for GPU to talk to Python on Windows 11 Home Edition and Dell Precision Workstation 5820

## In Summary

1. Install 2019 +C++ game.
2. Install the GPU drivers [NVIDIA CUDA version 11.7].
3. Installing cuDNN SDK (matching with CUDA).
4. Check if these paths are in the system's `Path` environment variable (no additional action required).

    ![System Paths](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3fdd2e78-8cd8-4292-a616-c885b0dbcbb7/Untitled.png)

5. Install Miniconda 3.9.
6. Create and activate a virtual environment using:

```bash
conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow-gpu<2.11"
```


## **For more information and troubleshooting, see below:**


## **Microsoft Visual Studio**

1. **Uninstall Visual Studio 2022**.
2. **Install Visual Studio 2019 +C++ game**.

### **Troubleshooting for VS 2022**

If you encounter issues with VS 2022, follow these steps:

1. Make sure C packages are installed.
2. Go to `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build`.
3. Copy the path and open it in the terminal.

    ![VC Build Path](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a487f927-a8b9-4938-87a5-8d3d251b7004/Untitled.png)

    ![VC Build Terminal](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07fa5603-cdf9-499d-9bdc-9424c48a83e3/Untitled.png)

## **NVIDIA GPU Drivers**

**Note**: Do not install version 12. cuDNN needs to match this version.

### **Checking for a CUDA-capable NVIDIA GPU**

To check if your system has a CUDA-capable NVIDIA GPU:

1. Check the specifications of your device given by the manufacturer.
2. Go to Windows **Device Manager → Display Adapters**.

    ![Display Adapters](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8df0b886-f419-455b-b63f-93762bc6216a/Untitled.png)

### **Installing the GPU Drivers [Install Version 11.7]**

If you don't have GPU drivers installed, download the correct driver from [NVIDIA's website](https://www.nvidia.com/download/index.aspx). Make sure to select the version 11.7.

### **Verifying the GPU Detection**

To verify the GPU detection:

```python
import tensorflow as tf
tf.test.gpu_device_name()
```

If you see the name of your GPU, TensorFlow has detected the NVIDIA GPU on your system.

## **cuDNN SDK**

### **Installing cuDNN SDK**

Make sure to install a cuDNN SDK version that matches the installed Python, TensorFlow, and CUDA Toolkit versions. Visit [this website](https://www.tensorflow.org/install/source_windows#gpu) to find the compatible cuDNN SDK version number.

1. Go to [NVIDIA's website](https://developer.nvidia.com/cudnn) to download the cuDNN SDK (you need an NVIDIA account).
2. [Download cuDNN v8.9.3 (July 11th, 2023), for CUDA 12.x](https://developer.nvidia.com/rdp/cudnn-download#a-collapse893-120).
3. Extract the downloaded .zip file and copy the necessary folders.
4. Paste the copied folders into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`.
5. Add these paths to the system's `Path` environment variable.

### **Troubleshooting**

If you encounter issues, you can try running the compiled examples from the CUDA Samples repository. Clone the project, build the samples, and run them using the instructions on the Github page.

## **Python**

Install Miniconda 3.9 from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

Check all the installed Python versions on Windows:

```bash
# Run this inside Anaconda prompt (Miniconda) or command prompt
(base) C:\Users\your_username>py -0
 -V:3.9 *         Python 3.9 (64-bit)

(base) C:\Users\your_username>py --list
 -V:3.9 *         Python 3.9 (64-bit)

(base) C:\Users\your_username>py -0p
 -V:3.9 *         C:\Users\your_username\AppData\Local\Programs\Python\Python39\python.exe
```

## **Install TensorFlow GPU**

Create and activate a virtual environment, then install TensorFlow GPU:

```bash
conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow-gpu<2.11"
```

## **Install Jupyter**

To install Jupyter, run the following command:

```bash
pip install jupyter
```

## **Verifying the GPU Detection**

After completing the installations, open a Python 3 Jupyter Notebook and execute the following command:

```python
import tensorflow as tf
print(len(tf.config.list_physical_devices('GPU')) > 0)
```

If it returns `True`, TensorFlow has detected the NVIDIA GPU on your system.

Congratulations! You have successfully set up TensorFlow 2 with GPU support on your Windows 11 Dell Workstation 5820. Enjoy your deep learning journey!

Sure, I have converted the provided text into Markdown for GitHub:

```markdown
## Python

### Install Miniconda 3.9

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4097dba1-3e72-4a3a-9bfa-e1e2a218d027/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b8d59737-ec66-4105-a09c-696eaec2a1a5/Untitled.png)

**Check all the installed Python versions on Windows**

```bash
# Run this inside Anaconda prompt (Miniconda) or Command Prompt
(base) C:\Users\peyma>py -0
 -V:3.9 *         Python 3.9 (64-bit)

(base) C:\Users\peyma>py --list
 -V:3.9 *         Python 3.9 (64-bit)

(base) C:\Users\peyma>py -0p
 -V:3.9 *         C:\Users\peyma\AppData\Local\Programs\Python\Python39\python.exe
```

Or inside the Anaconda prompt

```bash
(base) C:\Users\peyma>python
Python 3.9.17 (main, Jul  5 2023, 21:22:06) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Install TensorFlow GPU

1. **Prerequisite**: Launch Anaconda prompt (Miniconda) and run the following:

```bash
python.exe -m pip install --upgrade setuptools pip
```

Here's an example:

```bash
conda create --name tf python=3.9
conda activate tf

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow-gpu<2.11"
```

The installation was successful. Let's test it:

```bash
(tf) C:\Users\peyma>python
Python 3.9.17 (main, Jul  5 2023, 20:47:11) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(len(tf.config.list_physical_devices('GPU'))>0)
True
```

- **Troubleshooting**

If the `pip install tensorflow-gpu` command gives you an error related to Windows 3.7, try using venv:

```bash
python.exe -m pip install --upgrade setuptools pip
```

## Install Jupyter

### Prerequisite (some are optional, my suggestion is to run them anyway)

```bash
pip cache purge
pip install six 
conda install ipykernel
conda install -y jupyter 

pip install --upgrade ipython jupyter
python -m ipykernel install --user --name tf --display-name "Python 3.9 (tensorflow)"
```

To reuse, go to base and run:

```bash
python -m ipykernel install --user --name SINet --display-name "Python 3.6 (SINet)"
```

### Install Jupyter

```bash
conda install -y jupyter 
```

If running Jupyter Notebook returns an error, use the following (when you are trying to create a new notebook):

```bash
pip install cchardet
pip install --upgrade charset_normalizer
```

Open a Jupyter Notebook and select "New/Python 39 (tf)."

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c437d446-a18c-4f7e-a1d2-8de1ad6040da/Untitled.png)

### Test your notebook.

```python
# What version of Python do you have?
import sys
import tensorflow.keras
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print()
print(f"Python {sys.version}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
```

The output should be:

```plaintext
Tensor Flow Version: 2.10.1

Python 3.9.17 (main, Jul  5 2023, 21:22:06) [MSC v.1916 64 bit (AMD64)]
WARNING:tensorflow:From C:\Users\peyma\AppData\Local\Temp\ipykernel_12632\1746287398.py:15: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
GPU is available
```
```

