import os.path
from jacksung.ai.GeoAttX import GeoAttX_I, GeoAttX_P, Huayu
from jacksung.ai.utils.fy import get_agri_file_path
import numpy as np
from datetime import datetime,timedelta

if __name__ == '__main__':  # common setting
    predict_minutes = 15
    file_path = r'dataset'
    file_name = r'FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20241201231500_20241201232959_4000M_V0001.HDF'

    x1_path = 'models/GeoAttX_I_x1.pt'
    x4_path = 'models/GeoAttX_I_x4.pt'
    x12_path = 'models/GeoAttX_I_x12.pt'
    GP_path = 'models/GeoAttX_P.pt'
    HY_path = 'models/GeoAttX_M.pt'

    I_net = GeoAttX_I(file_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
    ys = I_net.predict(file_name, predict_minutes, p_steps=(4, 1))
    I_net.save(file_name, ys)
    # *******************************************************************************************************
    P_net = GeoAttX_P(GP_path, config='./configs/config_qpe.yml')
    M_net = Huayu(HY_path, config='./configs/config_imerg.yml')
    for y_date, y_np in ys.items():
        input_data_path = os.path.join(P_net.get_root_path(), 'input.npy')
        os.makedirs(P_net.get_root_path(), exist_ok=True)
        os.makedirs(M_net.get_root_path(), exist_ok=True)
        np.save(input_data_path, y_np)
        y_imerg = M_net.predict(input_data_path)
        y_qpe = P_net.predict(input_data_path)
        os.remove(input_data_path)
        print(y_imerg.shape)
        print(y_qpe.shape)
        if y_imerg is not None:
            M_net.save(y_imerg, y_date.strftime("%Y%m%d_%H%M%S"))
        if y_qpe is not None:
            P_net.save(y_qpe, y_date.strftime("%Y%m%d_%H%M%S"))
