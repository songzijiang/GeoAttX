# GeoAttX

The official code
of preprint paper: ["GeoAttX: A Novel Deep Learning Framework for Quarter-Hourly Precipitation Forecasting Using Geostationary Meteorological Satellite Observations"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5168620).

Any questions can be touched by jacksung1995@gmail.com

## pre-trained model

The pre-trained model can be downloaded from Google Drive:
[GeoAttX_I_x1](https://drive.google.com/file/d/1EcdUOjVr9veytt5NHGYJGkoNw7KVtMh3/view?usp=sharing)
[GeoAttX_I_x4](https://drive.google.com/file/d/119AyUk8-1m1eknRZoGWwbp3-xDi6t74C/view?usp=sharing)
[GeoAttX_I_x12](https://drive.google.com/file/d/1ohnmmrSsFZd_y04gg6-N9qOcAwWxtt2H/view?usp=sharing)
[GeoAttX_P](https://drive.google.com/file/d/1Mp9Qltk2eUbFzscL7s-rgnBwrQXzoS6z/view?usp=sharing)
[GeoAttX_M (Huayu)](https://drive.google.com/file/d/1klQ9sK0BjIciLSMxWPbfAxoD8-Cw06X4/view?usp=drive_link)

# How to run GeoAttX?

## step1, 
download the pre-trained model to folder "models".

-- notes, the file structure should be like this:
```
models/
    |-- GeoNet
    |-- GeoAttX_I_x1.pth
    |-- GeoAttX_I_x4.pth
    |-- GeoAttX_I_x12.pth
    |-- GeoAttX_P.pth
    |-- GeoAttX_M.pth
```

## step2, 
download FY-4B 4km full disk data and set the file path in "predict.py".

-- notes, the file structure should be like this:

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

## step3, 
set predicted minutes and the start file in "predict.py":

```python
predict_minutes = 180
file_path = 'data_dir'
file_name = 'FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240501001500_20240501002959_4000M_V0001.HDF'
```

## step4, 
run the code:

```bash
python predict.py
```

## Notes:
### GeoAttX_M (Huayu) training and validation data list.
The data list for GeoAttX_M can be found in the following format:
```{path_to_data}/{year}_{month}/{day}/{year}{month}{day}_{hour}_{minute}_{m_jdx}_{m_idx}.npy```
m_jdx: the middle index of the width.
m_idx: the middle index of the height.
### Contact us
Email: jacksung1995@gmail.com
