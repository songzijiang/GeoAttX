import os.path
from models.model_util.GeoAttX import GeoAttX_I, GeoAttX_P
import numpy as np
from configs.predict_config import predict_minutes, file_path, file_name, x1_path, x4_path, x12_path, model_path

if __name__ == '__main__':
    I_net = GeoAttX_I(file_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
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
