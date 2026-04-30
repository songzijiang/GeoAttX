# GeoAttX

Official quick-start code for the GeoAttX framework and Huayu.

This demo is based on PyTorch and requires a CUDA-capable GPU. The core model APIs are provided by the `jacksung` Python package.

## Requirements

- Python >= 3.11
- CUDA-capable GPU
- PyTorch
- `jacksung`

Install the utility package:

```bash
pip install jacksung
```

## Pre-trained Models

Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1a_F-BVZkql99m8Y_3xvFPWgyBPhm0zqV?usp=sharing).

Place them in the `models/` directory:

```text
models/
  GeoAttX_x1.pt
  GeoAttX_x4.pt
  GeoAttX_x12.pt
  GeoAttX_P.pt
  Huayu.pt
```

## Data Preparation

Download FY-4B AGRI 4 km full-disk data from the [NSMC Data Portal](https://satellite.nsmc.org.cn/DataPortal/cn/home/index.html), then set the data path in the corresponding config file under `configs/`.

You can also download the demo data from [Google Drive](https://drive.google.com/drive/folders/1KZP9R-ViOM5gEUCRrlcsBd0zaEgxKVhh?usp=sharing).

Place the demo data as follows:

```text
dataset/
  2024/
    9/
      16/
        FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916000000_20240916001459_4000M_V0001.HDF
        FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916001500_20240916002959_4000M_V0001.HDF
        ...
```

The repository also includes normalization/statistics files in `dataset/`, which are required by the demo scripts.

## Run Prediction

Edit `predict.py` to set the forecast lead time and start time:

```python
predict_minutes = 15
current_date = datetime(year=2024, month=9, day=16, hour=0, minute=15)
```

Then run:

```bash
python predict.py
```

`predict.py` runs `GeoAttX_I` (`GeoAttX`) first, then feeds the prediction into `GeoAttX_P` for precipitation estimation.

Make sure the FY-4B files for the current time and required previous time steps exist. For example, to predict from `2024-09-16 00:30` with `predict_minutes = 15`, the required preceding satellite files must be available. See the RGA section in the paper for details.

`GeoAttX_P` outputs precipitation stands the 15-minute precipitation `mm`.
## Batch Prediction

Use `range_predict.py` to run prediction over a time range:

```python
start_date = datetime(year=2024, month=9, day=16, hour=0, minute=15)
end_date = datetime(year=2024, month=9, day=16, hour=0, minute=30)
```

Run:

```bash
python range_predict.py
```

The script uses `MultiTasks(8)` by default. Adjust the worker count according to your GPU memory and runtime environment.

## Run Huayu Only

Edit the satellite file path in `Huayu.py`, then run:

```bash
python Huayu.py
```

Huayu outputs precipitation in `mm/hhr`. To obtain 15-minute precipitation, use `hy / 2`.

## Configuration Files

- `configs/config_predict.yml`: `GeoAttX_I` prediction settings
- `configs/config_qpe.yml`: `GeoAttX_P` precipitation estimation settings
- `configs/config_imerg.yml`: Huayu settings

Common options:

- `save_path`: output directory, default `./results`
- `gpu_ids`: GPU indices, for example `[0]`
- `pred_data_path` / `prec_data_path`: input data root used by the corresponding task

Do not change the model architecture parameters unless you also use matching checkpoints.

## Results

Outputs are saved to `results/`.

Filename prefixes:

- `pred*`: `GeoAttX_I` prediction results
- `prec*`: `GeoAttX_P` precipitation results
- `prem*`: Huayu precipitation results

## Change Target Area

Set `area` when initializing `GeoAttX_I`:

```python
I_net = GeoAttX_I(
    norm_path,
    x1_path,
    x4_path,
    x12_path,
    config="./configs/config_predict.yml",
    area=((100, 140, 10), (20, 60, 10)),
)
```

This means longitude `100E` to `140E` and latitude `20N` to `60N`.

Notes:

- Keep the step size as `10`
- Longitude and latitude ranges should each span `40`
- Valid area bounds are `45E` to `165E` and `60S` to `60N` (decided by FY-4B's coverage)

## GeoAttX_M (Huayu) Data List

The training and validation lists for `GeoAttX_M` are in:

```text
dataset/GeoAttX_M/train.txt
dataset/GeoAttX_M/test.txt
```

Format:

```text
{path_to_data}/{year}_{month}/{day}/{year}{month}{day}_{hour}_{minute}_{m_jdx}_{m_idx}.npy
```

where:

- `m_jdx`: middle index of width
- `m_idx`: middle index of height

## Contact

Email: jacksung1995@gmail.com
