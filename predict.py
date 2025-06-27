import os.path
from jacksung.ai.GeoAttX import GeoAttX_I, GeoAttX_P, GeoAttX_M
import numpy as np
from configs.predict_config import predict_minutes, file_path, file_name, x1_path, x4_path, x12_path, model_path, \
    imerg_path

if __name__ == '__main__':
    I_net = GeoAttX_I(file_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
    ys = I_net.predict(file_name, predict_minutes, p_steps=(4, 1))
    I_net.save(file_name, ys)
    # *******************************************************************************************************
    P_net = GeoAttX_P(model_path, config='./configs/config_qpe.yml')
    M_net = GeoAttX_M(imerg_path, config='./configs/config_imerg.yml')
    for y_date, y_np in ys.items():
        input_data_path = os.path.join(P_net.get_root_path(), 'input.npy')
        os.makedirs(P_net.get_root_path(), exist_ok=True)
        os.makedirs(M_net.get_root_path(), exist_ok=True)
        np.save(input_data_path, y_np)
        y_imerg = M_net.predict(input_data_path)
        y_qpe = P_net.predict(input_data_path)
        os.remove(input_data_path)
        if y_imerg is not None:
            P_net.save(y_imerg, y_date.strftime("%Y%m%d_%H%M%S"))
        if y_qpe is not None:
            P_net.save(y_qpe, y_date.strftime("%Y%m%d_%H%M%S"))
