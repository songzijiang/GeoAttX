# GeoAttX

The official quick start code of GeoAttX framework and Huayu. This demo code is based on PyTorch and requires GPU. 
# How to run Huayu/GeoAttX?

## Step1, 
Download the pre-trained model to the folder "models".

The pre-trained model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1a_F-BVZkql99m8Y_3xvFPWgyBPhm0zqV?usp=sharing)

-- notes, the file tree should be like this:
```
models/
    |-- GeoAttX_x1.pth
    |-- GeoAttX_x4.pth
    |-- GeoAttX_x12.pth
    |-- GeoAttX_P.pth
    |-- Huayu.pth
```

And install the util package jacksung (require Python version >=3.11) by:
```
pip install jacksung
```

## Step2, 
Download FY-4B 4km full disk data from [here](https://satellite.nsmc.org.cn/DataPortal/cn/home/index.html) and set the data path in "config_*.yml".

or

download demo data from [here](https://drive.google.com/drive/folders/1KZP9R-ViOM5gEUCRrlcsBd0zaEgxKVhh?usp=sharing) put in the folder "dataset/2024/9/16".
-- notes, the file tree should be like this:

```
dataset/
    |-- 2024/
        |-- 9/
            |-- 16/
                |-- FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916000000_20240916001459_4000M_V0001.HDF
                |-- FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916001500_20240916002959_4000M_V0001.HDF
                |-- ...
                |-- ...
        |-- ...
```

## Step3, 
Set predicted minutes and the start file:

```python
from datetime import datetime
predict_minutes = 15
current_date = datetime(year=2024, month=9, day=16, hour=0, minute=15)
```

if you only want to use Huayu, you could set the "predict_minutes" to zero.
```python
predict_minutes = 0
```
Note: if you want to predict a result, make sure the files of current date and the previous date are existing.

e.g., predicting 2024-09-16 00:30 from current date 2024-09-16 00:15 with 'predict_minutes=15', you need to make sure the files of 2024-09-16 00:15 and 2024-09-16 00:00 are existing. More details please refer to RGA in paper.


## Step4, 
Run the code:

```bash
python predict.py
```

or run the batch code:

```bash
python range_predict.py
```

please change the number of the threads according to your computer performance:
```python
mt = MultiTasks(8)
```

## Step5, 
The results will be produced in the "results" folder

'pred*' denotes GeoAttX model results
'prec*' denotes GeoAttX_P model results
'prem*' denotes Huayu model results

## Notes:
### How to chang the target area?
set the 'area' in
```python
I_net = GeoAttX_I(norm_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml',area=((100, 140, 10), (20, 60, 10)))
```
'((100, 140, 10), (20, 60, 10))' means from longitude 100°E to 140°E, and from latitude 20°N to 60°N. Please do not change the step size (10).
The range of both longtitude and latitude should be '40' and in the area from 45°E to 165°E and from -60°S to 60°N.
### Huayu (GeoAttX_M) training and validation data list.
The data list for GeoAttX_M can be found in 'dataset/GeoAttX_M' in the following format:
```{path_to_data}/{year}_{month}/{day}/{year}{month}{day}_{hour}_{minute}_{m_jdx}_{m_idx}.npy```
where "m_jdx" means the middle index of the width and "m_idx" means the middle index of the height.
### Contact us
Email: jacksung1995@gmail.com
