import os.path
from jacksung.ai.GeoAttX import GeoAttX_I, GeoAttX_P, Huayu
from jacksung.ai.utils.fy import get_agri_file_path
import numpy as np
from datetime import datetime, timedelta

if __name__ == '__main__':  # common setting
    predict_minutes = 15
    norm_path = r'dataset'

    x1_path = 'models/GeoAttX_x1.pt'
    x4_path = 'models/GeoAttX_x4.pt'
    x12_path = 'models/GeoAttX_x12.pt'
    GP_path = 'models/GeoAttX_P.pt'

    current_date = datetime(year=2024, month=9, day=16, hour=0, minute=30)
    I_net = GeoAttX_I(norm_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
    ys = I_net.predict(current_date, predict_minutes, p_steps=(4, 1))
    I_net.save(current_date, ys)
    # *******************************************************************************************************
    P_net = GeoAttX_P(norm_path, GP_path, config='./configs/config_qpe.yml')
    for y_date, y_np in ys.items():
        input_data_path = os.path.join(P_net.get_root_path(), 'input.npy')
        os.makedirs(P_net.get_root_path(), exist_ok=True)
        np.save(input_data_path, y_np)
        y_qpe = P_net.predict(input_data_path)
        os.remove(input_data_path)
        print(y_qpe.shape)
        if y_qpe is not None:
            P_net.save(y_qpe, y_date.strftime("%Y%m%d_%H%M%S"))
