import os.path
from jacksung.ai.GeoAttX import GeoAttX_I, GeoAttX_P, Huayu
import numpy as np
from datetime import datetime, timedelta

if __name__ == '__main__':
    # Predict the range of dates
    predict_minutes = 180
    file_path = r'dataset'

    x1_path = 'models/GeoAttX_I_x1.pt'
    x4_path = 'models/GeoAttX_I_x4.pt'
    x12_path = 'models/GeoAttX_I_x12.pt'
    GP_path = 'models/GeoAttX_P.pt'
    HY_path = 'models/GeoAttX_M.pt'

    # range_predict.py setting
    start_date = datetime(year=2024, month=9, day=15, hour=0, minute=0) - timedelta(hours=8)
    end_date = datetime(year=2024, month=9, day=21, hour=2, minute=59) - timedelta(hours=8)

    file_names = []
    current_date = start_date
    I_net = GeoAttX_I(file_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
    M_net = Huayu(HY_path, config='./configs/config_imerg.yml')
    while current_date <= end_date:
        start_str = current_date.strftime("%Y%m%d%H%M%S")
        end_str = (current_date + timedelta(minutes=14, seconds=59)).strftime("%Y%m%d%H%M%S")
        file_names.append(rf'FY4B-_AGRI--_N_DISK_{105 if current_date > datetime(year=2023, month=3, day=1) else 133}' +
                          rf'0E_L1-_FDI-_MULT_NOM_{start_str}_{end_str}_4000M_V0001.HDF')
        current_date += timedelta(minutes=15)
    for idx, file_name in enumerate(file_names):
        print(rf'{idx}/{len(file_names)}, 预测起始文件：{file_name}')
        I_net.set_root_path()
        ys = I_net.predict(file_name, predict_minutes, p_steps=(4, 1))
        I_net.save(file_name, ys)
        # *******************************************************************************************************
        for y_date, y_np in ys.items():
            input_data_path = os.path.join(I_net.get_root_path(), 'input.npy')
            os.makedirs(M_net.get_root_path(), exist_ok=True)
            np.save(input_data_path, y_np)
            y = M_net.predict(input_data_path)
            os.remove(input_data_path)
            if y is not None:
                M_net.save(y, y_date.strftime("%Y%m%d_%H%M%S"))
        current_date += timedelta(minutes=15)
        print('*' * 60)
