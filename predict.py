import os.path
from models.model_util.GeoAttX import GeoAttX_I, GeoAttX_P
import numpy as np

if __name__ == '__main__':
    x1_path = 'models/GeoAttX_I_x1.pt'
    x4_path = 'models/GeoAttX_I_x4.pt'
    x12_path = 'models/GeoAttX_I_x12.pt'
    predict_minutes = 180
    file_path = r'data_dir'
    file_name = r'FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20241201001500_20241201002959_4000M_V0001.HDF'
    I_net = GeoAttX_I(file_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
    ys = I_net.predict(file_name, predict_minutes, p_steps=(4, 1))
    I_net.save(file_name, ys)
    # *******************************************************************************************************
    model_path = 'models/GeoAttX_P.pt'
    net = GeoAttX_P(model_path, config='./configs/config_qpe.yml')
    for y_date, y_np in ys.items():
        input_data_path = os.path.join(net.get_root_path(), 'input.npy')
        os.makedirs(net.get_root_path(), exist_ok=True)
        np.save(input_data_path, y_np)
        y = net.predict(input_data_path)
        os.remove(input_data_path)
        if y is not None:
            net.save(y, y_date.strftime("%Y%m%d_%H%M%S"))
