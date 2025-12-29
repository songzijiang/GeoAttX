# GeoAttX

The official code
of preprint paper: ["Huayu: Advanced Real-Time Precipitation Estimation from Geostationary Satellite"]() and ["GeoAttX: A Novel Framework for Cloud Image Forecasting Using Geostationary Satellite Observations"]()

# How to run Huayu/GeoAttX?

## Step1, 
Download the pre-trained model to the folder "models".

The pre-trained model can be downloaded from Google Drive:

1.[Huayu (GeoAttX_M)](https://drive.google.com/file/d/1klQ9sK0BjIciLSMxWPbfAxoD8-Cw06X4/view?usp=drive_link)
2.[GeoAttX_I_x1](https://drive.google.com/file/d/1EcdUOjVr9veytt5NHGYJGkoNw7KVtMh3/view?usp=sharing)
3.[GeoAttX_I_x4](https://drive.google.com/file/d/119AyUk8-1m1eknRZoGWwbp3-xDi6t74C/view?usp=sharing)
4.[GeoAttX_I_x12](https://drive.google.com/file/d/1ohnmmrSsFZd_y04gg6-N9qOcAwWxtt2H/view?usp=sharing)
5.[GeoAttX_P](https://drive.google.com/file/d/1Mp9Qltk2eUbFzscL7s-rgnBwrQXzoS6z/view?usp=sharing)

-- notes, the file tree should be like this:
```
models/
    |-- GeoNet
    |-- GeoAttX_I_x1.pth
    |-- GeoAttX_I_x4.pth
    |-- GeoAttX_I_x12.pth
    |-- GeoAttX_P.pth
    |-- GeoAttX_M.pth
```

And install the util package jacksung by:
```
pip install jacksung
```

## Step2, 
Download FY-4B 4km full disk data from [here](https://satellite.nsmc.org.cn/DataPortal/cn/home/index.html) and set the file path in "predict.py".

-- notes, the file tree should be like this:

```
data_dir/
    |-- 2024/
        |-- 04/
            |-- 01/
                |-- FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240401000000_20240401001459_4000M_V0001.HDF
                |-- FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240401001500_20240401002959_4000M_V0001.HDF
                |-- ...
        |-- 05/
            |-- 01/
                |-- FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240501000000_20240501001459_4000M_V0001.HDF
                |-- FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240501001500_20240501002959_4000M_V0001.HDF
                |-- ...
        |-- ...
```

## Step3, 
Set predicted minutes and the start file:

```python
from datetime import datetime
predict_minutes = 180
current_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)
```

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
the results will be produced in the "result" folder

## Notes:
### Huayu (GeoAttX_M) training and validation data list.
The data list for GeoAttX_M can be found in 'dataset/GeoAttX_M' in the following format:
```{path_to_data}/{year}_{month}/{day}/{year}{month}{day}_{hour}_{minute}_{m_jdx}_{m_idx}.npy```
where "m_jdx" means the middle index of the width and "m_idx" means the middle index of the height.
### Contact us
Email: jacksung1995@gmail.com
