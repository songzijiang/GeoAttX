import os.path
from jacksung.ai.GeoAttX import GeoAttX_I, GeoAttX_P, Huayu
from datetime import datetime, timedelta

if __name__ == '__main__':  # common setting

    norm_path = r'dataset'
    HY_path = 'models/Huayu.pt'

    current_date = datetime(year=2024, month=9, day=16, hour=0, minute=15)
    M_net = Huayu(norm_path, HY_path, config='./configs/config_imerg.yml')
    os.makedirs(M_net.get_root_path(), exist_ok=True)
    hy = M_net.predict(
        satellite_file=rf'dataset/2024/9/16/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916000000_20240916001459_4000M_V0001.HDF')
    print(hy.shape)
    if hy is not None:
        M_net.save(hy, current_date.strftime("%Y%m%d_%H%M%S"))
