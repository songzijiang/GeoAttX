import os.path
from jacksung.ai.GeoAttX import GeoAttX_I, GeoAttX_P, Huayu
import numpy as np
from datetime import datetime, timedelta
from jacksung.utils.multi_task import MultiTasks
from jacksung.utils.time import Stopwatch


def multi_worker(current_date):
    st = Stopwatch()
    I_net = GeoAttX_I(norm_path, x1_path, x4_path, x12_path, config='./configs/config_predict.yml')
    M_net = Huayu(norm_path, HY_path, config='./configs/config_imerg.yml')
    print(rf'预测起始时间：{current_date.strftime("%Y-%m-%d %H:%M:%S")}')
    I_net.set_root_path()
    M_net.set_root_path()
    ys = I_net.predict(current_date, predict_minutes, p_steps=(4, 1))
    I_net.save(current_date, ys, save_target=False)
    # *******************************************************************************************************
    for y_date, y_np in ys.items():
        if y_date != list(ys.keys())[-1]:
            continue
        input_data_path = os.path.join(I_net.get_root_path(), 'input.npy')
        os.makedirs(M_net.get_root_path(), exist_ok=True)
        np.save(input_data_path, y_np)
        y = M_net.predict(input_data_path)
        os.remove(input_data_path)
        if y is not None:
            M_net.save(y, y_date.strftime("%Y%m%d_%H%M%S"))
    print(rf'{current_date.strftime("%Y-%m-%d %H:%M:%S")}预测完成，耗时{st.pinch()}')


if __name__ == '__main__':
    # Predict the range of dates
    predict_minutes = 180
    norm_path = r'dataset'

    x1_path = 'models/GeoAttX_x1.pt'
    x4_path = 'models/GeoAttX_x4.pt'
    x12_path = 'models/GeoAttX_x12.pt'
    GP_path = 'models/GeoAttX_P.pt'
    HY_path = 'models/Huayu.pt'

    # range_predict.py setting
    start_date = datetime(year=2024, month=9, day=15, hour=15, minute=0)
    end_date = datetime(year=2024, month=9, day=17, hour=23, minute=59)

    current_date = start_date
    mt = MultiTasks(8)
    while current_date <= end_date:
        # single thread
        # multi_worker(current_date)
        # multi threads
        mt.add_task(current_date, multi_worker, [current_date])
        current_date += timedelta(minutes=15)
    mt.execute_task()
