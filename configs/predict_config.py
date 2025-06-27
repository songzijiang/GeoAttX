from datetime import datetime, timedelta

# common setting
predict_minutes = 180
file_path = r'/mnt/data1/szj/FY/downloaded_file'
file_name = r'FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20241201001500_20241201002959_4000M_V0001.HDF'

x1_path = 'models/GeoAttX_I_x1.pt'
x4_path = 'models/GeoAttX_I_x4.pt'
x12_path = 'models/GeoAttX_I_x12.pt'
model_path = 'models/GeoAttX_P.pt'
imerg_path = 'models/GeoAttX_M.pt'

# range_predict.py setting
start_date = datetime(year=2025, month=6, day=7, hour=0, minute=0)-timedelta(hours=8)
end_date = datetime(year=2025, month=6, day=8, hour=23, minute=59)-timedelta(hours=8)
