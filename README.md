pythonDicomConverter
============
A simple DICOM converter to 3D volume data.

<img src='https://raw.githubusercontent.com/ChenglongWang/pythonDicomConverter/master/mainview.PNG' align='right' width='270'>

pythonDicomConverter is an simple open source DICOM file converter to 3D volume data (Raw and Nifti format). It enables a easy and quick converting by pre-defined rules.

pythonDicomConverter is separated from [dicompyler](https://github.com/bastula/dicompyler) which is designed for radiation therapy research. Same as dicompyler, pythonDicomConverter is a wxPython-based application written in Python.


Getting Started:
----------------
### Install

pythonDicomConverter can be easily installed using pip command. Dependency packages will be automatically installed. 

`pip install git+https://github.com/ChenglongWang/pythonDicomConverter.git --process-dependency-links`

### Run

After installing the package, a executable script will be present on your path. Now you can run a simple command `dcm-cvt` from your terminal. 

### How to use

1. Choose a folder containing DICOM files. Click `Scan` button to scan files.
2. Adjust convert options to fit your demands. The target series will be highlighted in red.
3. Ensure the output folder and file name are correct.
4. Click `Convert` button to process converting.



