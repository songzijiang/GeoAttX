import os.path
from models.model_util.GeoAttX import GeoAttX_I, GeoAttX_P
import numpy as np
from configs.predict_config import predict_minutes, file_path, file_name, x1_path, x4_path, x12_path, model_path, \
    start_date, end_date
from datetime import datetime, timedelta

if __name__ == '__main__':
    # Predict the range of dates
    file_names = []
    current_date = start_date
    while current_date <= end_date:
        start_str = current_date.strftime("%Y%m%d%H%M%S")
        end_str = (current_date + timedelta(minutes=14, seconds=59)).strftime("%Y%m%d%H%M%S")
        file_names.append(rf'FY4B-_AGRI--_N_DISK_{105 if current_date > datetime(year=2025, month=3, day=1) else 133}' +
                          rf'0E_L1-_FDI-_MULT_NOM_{start_str}_{end_str}_4000M_V0001.HDF')
        current_date += timedelta(minutes=15)
    for idx, file_name in enumerate(file_names):
        I_net = GeoAttX_I(file_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
        print(rf'{idx}/{len(file_names)}, {file_name}')
        ys = I_net.predict(file_name, predict_minutes, p_steps=(4, 1))
        I_net.save(file_name, ys)
        # *******************************************************************************************************
        net = GeoAttX_P(model_path, config='./configs/config_qpe.yml')
        for y_date, y_np in ys.items():
            input_data_path = os.path.join(net.get_root_path(), 'input.npy')
            os.makedirs(net.get_root_path(), exist_ok=True)
            np.save(input_data_path, y_np)
            y = net.predict(input_data_path)
            os.remove(input_data_path)
            if y is not None:
                net.save(y, y_date.strftime("%Y%m%d_%H%M%S"))
        current_date += timedelta(minutes=15)
        print('*' * 20)
