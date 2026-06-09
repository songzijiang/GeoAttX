import os.path
from jacksung.ai.GeoAttX import Huayu

if __name__ == '__main__':
    # common setting
    norm_path = r'dataset'
    HY_path = 'models/Huayu.pt'
    config = './configs/config_imerg.yml'
    # set up the longitude and latitude: from 100° E to 140° E and from 20° N to 60° N.
    area = ((100, 140, 10), (20, 60, 10))
    M_net = Huayu(norm_path, HY_path, config=config, print_timelog=True, area=area)
    os.makedirs(M_net.get_root_path(), exist_ok=True)
    hy = M_net.predict(
        satellite_file=rf'dataset/2024/9/16/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916000000_20240916001459_4000M_V0001.HDF')
    # hy stands for the precipitation (mm/hhr), if you want to get the precipitation in 15 minutes, please ues "hy / 2".
    print(hy.shape)
    if hy is not None:
        M_net.save(hy / 2, 'Huayu')
