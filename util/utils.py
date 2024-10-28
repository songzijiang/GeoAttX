import sys

sys.path.append('../')
import torch
import torch.nn as nn
import math
import cv2
import shutil
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import os
from jacksung.utils.data_convert import nc2np, np2tif
from jacksung.utils.image import crop_png, zoom_image, zoomAndDock
from jacksung.utils.cache import Cache
import rasterio
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from rasterio.transform import from_origin
import yaml
import argparse
from jacksung.utils.multi_task import type_thread, type_process, MultiTasks
import jacksung.utils.fastnumpy as fnp
from tqdm import tqdm
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator
from models.GeoNet.m_networkV2 import GeoNet
from models.unet.unet_model import UNet
from models.swinir.network_swinir import SwinIR
import netCDF4 as nc
import ephem
import math
from jacksung.utils.data_convert import np2tif, Coordinate

x_range = {'left': -60, 'top': 60, 'bottom': -60, 'right': 60, 'width': 2400, 'height': 2400}
min_x_range = {'left': -60, 'top': 60, 'bottom': -60, 'right': 60, 'width': 480, 'height': 480}

import cv2


def getFY_coord(ld):
    return Coordinate(left=ld + x_range['left'], top=x_range['top'], right=ld + x_range['right'],
                      bottom=x_range['bottom'], h=x_range['height'], w=x_range['width'])
    # return Coordinate(left=ld - 45, top=36, right=ld + 45, bottom=-36, h=1571, w=1963)


def getFY_coord_min(ld):
    return Coordinate(left=ld + min_x_range['left'], top=min_x_range['top'], right=ld + min_x_range['right'],
                      bottom=min_x_range['bottom'], h=min_x_range['height'], w=min_x_range['width'])


def getFY_coord_clip():
    return Coordinate(left=100, top=60, right=140, bottom=20, h=800, w=800)


static_params = {4000: {'l': 2747, 'c': 2747, 'COFF': 1373.5, 'CFAC': 10233137, 'LOFF': 1373.5, 'LFAC': 10233137},
                 2000: {'l': 5495, 'c': 5495, 'COFF': 2747.5, 'CFAC': 20466274, 'LOFF': 2747.5, 'LFAC': 20466274},
                 1000: {'l': 10991, 'c': 10991, 'COFF': 5495.5, 'CFAC': 40932549, 'LOFF': 5495.5, 'LFAC': 40932549},
                 500: {'l': 21983, 'c': 21983, 'COFF': 10991.5, 'CFAC': 81865099, 'LOFF': 10991.5, 'LFAC': 81865099},
                 250: {'l': 43967, 'c': 43967, 'COFF': 21983.5, 'CFAC': 163730199, 'LOFF': 21983.5, 'LFAC': 163730199}}


# FY4星下点行列号转经纬度
def xy2coordinate(l, c, ld=105, res=4000):
    ea = 6378.137
    eb = 6356.7523
    h = 42164
    # 4000m分辨率
    COFF = static_params[res]['COFF']
    CFAC = static_params[res]['CFAC']
    LOFF = static_params[res]['LOFF']
    LFAC = static_params[res]['LFAC']

    x = (np.pi * (c - COFF)) / (180 * (2 ** -16) * CFAC)
    y = (np.pi * (l - LOFF)) / (180 * (2 ** -16) * LFAC)

    sd_t1 = np.square(h * np.cos(x) * np.cos(y))
    sd_t2 = np.square(np.cos(y)) + np.square(ea) / np.square(eb) * np.square(np.sin(y))
    sd_t3 = np.square(h) - np.square(ea)
    sd = np.sqrt(sd_t1 - sd_t2 * sd_t3)
    sn = (h * np.cos(x) * np.cos(y) - sd) / (np.cos(y) ** 2 + ea ** 2 / eb ** 2 * np.sin(y) ** 2)

    s1 = h - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1 ** 2 + s2 ** 2)

    lon = 180 / np.pi * np.arctan(s2 / s1) + ld
    lat = 180 / np.pi * np.arctan(ea ** 2 / eb ** 2 * s3 / sxy)
    return lon, lat


def get_model(args, task_type=None):
    # definitions of model
    if args.model in ['geonet']:
        model = GeoNet(window_sizes=args.window_sizes, n_lgab=args.n_lgab, c_in=args.c_in, c_lgan=args.c_lgan,
                       r_expand=args.r_expand, down_sample=args.down_sample, num_heads=args.num_heads,
                       task=task_type if task_type else args.task, downstage=args.downstage)
    elif args.model == 'unet':
        model = UNet(c_in=args.c_in)
    elif args.model == 'swinir':
        window_size = 10
        height = (1040 // args.down_sample // window_size + 1) * window_size
        width = (1600 // args.down_sample // window_size + 1) * window_size
        model = SwinIR(upscale=args.down_sample, img_size=(height, width), in_chans=4,
                       window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                       embed_dim=108, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    elif args.model == 'deeplab':
        model = None
    else:
        model = None
    if args.fp == 16:
        model.half()
    elif args.fp == 64:
        model.double()
    return model


def load_model(model, state_dict, strict=True):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        name = name[name.index('.') + 1:]
        if name in own_state.keys():
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                # own_state[name].requires_grad = False
            except Exception as e:
                err_log = f'While copying the parameter named {name}, ' \
                          f'whose dimensions in the model are {own_state[name].size()} and ' \
                          f'whose dimensions in the checkpoint are {param.size()}.'
                if not strict:
                    print(err_log)
                else:
                    raise Exception(err_log)
        elif strict:
            raise KeyError(f'unexpected key {name} in {own_state.keys()}')
        else:
            print(f'{name} not loaded by model')


def get_stat_dict(metrics):
    stat_dict = {
        'epochs': 0, 'losses': [], 'ema_loss': 0.0, 'metrics': {}}
    for idx, metrics in enumerate(metrics):
        name, default_value, op = metrics
        stat_dict['metrics'][name] = {'value': [], 'best': {'value': default_value, 'epoch': 0, 'op': op}}
    return stat_dict


def data_to_device(datas, device, fp=32):
    outs = []
    for data in datas:
        if fp == 16:
            data = data.type(torch.HalfTensor)
        if fp == 64:
            data = data.type(torch.DoubleTensor)
        if fp == 32:
            data = data.type(torch.FloatTensor)
        data = data.to(device)
        outs.append(data)
    return outs


def draw_lines(yaml_path):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print('[TemporaryTag]Producing the LinePicture of Log...', end='[TemporaryTag]\n')
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    # 创建图表
    m_len = len(yaml_args['metrics'])
    plt.figure(figsize=(10 * (m_len + 1), 6))  # 设置图表的大小
    x = np.array(range(1, yaml_args['epochs'] + 1))
    for idx, d in enumerate(yaml_args['metrics'].items()):
        m_name, m_value = d
        y = np.array(m_value['value'])
        # 生成数据
        plt.subplot(1, m_len + 1, idx + 1)
        plt.plot(x, y)
        plt.title(m_name)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    y = np.array(yaml_args['losses'])
    plt.subplot(1, m_len + 1, m_len + 1)
    scale = len(y) / yaml_args['epochs']
    x = np.array(range(1, len(y) + 1)) / scale
    plt.plot(x, y)
    plt.title('Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # 添加图例
    # plt.legend()
    plt.savefig(os.path.join(os.path.dirname(yaml_path), 'Metrics.jpg'))


def make_best_metric(stat_dict, metrics, epoch, save_model_param, server_log_param):
    save_model_flag = False
    experiment_model_path, model, optimizer, scheduler = save_model_param
    log, epochs, cloudLogName = server_log_param

    for name, m_value in metrics:
        stat_dict['metrics'][name]['value'].append(m_value)
        inf = float('inf')
        if eval(str(m_value) + stat_dict['metrics'][name]['best']['op'] + str(
                stat_dict['metrics'][name]['best']['value'])):
            stat_dict['metrics'][name]['best']['value'] = m_value
            stat_dict['metrics'][name]['best']['epoch'] = epoch
            log.send_log('{}:{} epoch:{}/{}'.format(name, m_value, epoch, epochs), cloudLogName)
            save_model_flag = True

    if save_model_flag:
        # sava best model
        save_model(os.path.join(experiment_model_path, 'model_{}.pt'.format(epoch)), epoch,
                   model, optimizer, scheduler, stat_dict)
    # '[Validation] nRMSE/RMSE: {:.4f}/{:.4f} (Best: {:.4f}/{:.4f}, Epoch: {}/{})\n'
    test_log = f'[Val epoch:{epoch}] ' \
               + '/'.join([str(m[0]) for m in metrics]) \
               + ': ' \
               + '/'.join([str(round(m[1], 4)) for m in metrics]) \
               + ' (Best: ' \
               + '/'.join([str(round(stat_dict['metrics'][m[0]]['best']['value'], 4)) for m in metrics]) \
               + ', Epoch: ' \
               + '/'.join([str(stat_dict['metrics'][m[0]]['best']['epoch']) for m in metrics]) \
               + ')'
    save_model(os.path.join(experiment_model_path, 'model_latest.pt'), epoch, model, optimizer, scheduler, stat_dict)
    return test_log


def save_np2file(data_list, name_lists, save_path):
    progress_bar = tqdm(total=len(data_list), desc='saving npy')
    for idx, data in enumerate(data_list):
        if name_lists:
            name = name_lists[idx]
        else:
            name = idx
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, str(name).replace(' ', '-') + '.npy'), data)
        progress_bar.update(1)


def save_model(_path, _epoch, _model, _optimizer, _scheduler, _stat_dict):
    # torch.save(model.state_dict(), saved_model_path)
    torch.save({
        'epoch': _epoch,
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': _optimizer.state_dict(),
        'scheduler_state_dict': _scheduler.state_dict(),
        'stat_dict': _stat_dict
    }, _path)


def parse_config(config=None):
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config', type=str, default=None, help='pre-config file for training')
    args = parser.parse_args()
    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(yaml_args)
    else:
        opt = vars(args)
        yaml_args = yaml.load(open(config), Loader=yaml.FullLoader)
        opt.update(yaml_args)

    # set visible gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpu_ids])

    # select active gpu devices
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        # device = torch.device('cuda:' + str(args.gpu_ids[0]))
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    return device, args


def read_nc(file_path, keys):
    dataset = nc.Dataset(file_path, 'r')  # 'r' 表示只读模式
    # print(dataset.variables.keys())  # 打印所有变量的名称
    numpy_out = None
    for key in keys:
        variable_data = dataset.variables[key]  # 读取变量数据
        # 将NetCDF变量数据转换为NumPy数组
        np_data = np.array(variable_data)
        if len(np_data.shape) == 3:
            np_data = np_data[:, np.newaxis, :, :]
        elif len(np_data.shape) == 4:
            np_data = np_data[:, np.newaxis, :, :, :]
        if numpy_out is None:
            numpy_out = np_data
        else:
            numpy_out = np.concatenate([numpy_out, np_data], axis=1)
    dataset.close()
    return numpy_out


def convert_file2idx(file_name):
    file_name = file_name.replace('.npy', '')
    h, m = file_name[:2], file_name[2:4]
    return int(h) * 4 + int(m) // 15


def get_reference(ld):
    # 构造控制点列表 gcps_list
    gcps_list = []
    step = 50
    last_corrd = []
    lc_list = []
    latlon_list = []
    for l in range(0, 2748, step):
        for c in range(0, 2748, step):
            lon, lat = xy2coordinate(l, c, ld=ld)
            if str(lon) == 'nan' or str(lat) == 'nan':
                continue
            skip_flag = False
            for corrd in last_corrd:
                if (corrd[0] - lon) ** 2 + (corrd[1] - lat) ** 2 <= 100:
                    skip_flag = True
                    break
            if skip_flag:
                continue
            last_corrd.append([lon, lat])
            gcps_list.append(gdal.GCP(lon, lat, 0, c, l))
            lc_list.append((l, c))
            latlon_list.append((lon, lat))
    # 设置空间参考
    # print('控制点数目：', len(gcps_list))
    # print([(l, c) for l, c in lc_list])
    # print([(lon, lat) for lon, lat in latlon_list])
    spatial_reference = osr.SpatialReference()
    spatial_reference.SetWellKnownGeogCS('WGS84')
    return spatial_reference, gcps_list


def convert_idx2file(file_name):
    file_name = file_name.replace('.npy', '')
    idx = int(file_name)
    h, m = str(idx // 4), str(idx % 4 * 15)
    if len(h) <= 1:
        h = '0' + h
    if len(m) <= 1:
        m = '0' + m
    return h + m


reference_cache = Cache(10)


def getNPfromHDFClip(ld, file_path, file_type='FDI', lock=None):
    d = (ld - 120) * 20
    np_data = getNPfromHDF(file_path, file_type, lock)
    np_data = np_data[:, 0:800, 800 - d:1600 - d]
    return np_data


def getNPfromHDF(hdf_path, file_type='FDI', lock=None):
    file_name = hdf_path.split(os.sep)[-1]
    file_info = prase_filename(file_name)
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    if file_type == 'FDI':
        f = ds.groups['Data']
        np_data = np.zeros((15, 2748, 2748), dtype=np.float32)
        for i in range(1, 16):
            s_i = '0' + str(i) if i < 10 else str(i)
            data = np.array(f[f'NOMChannel{s_i}'][:]).astype(np.float32)
            data[data > 10000] = np.nan
            np_data[i - 1] = data
        # np_data = np_data[6:15]
        in_out_idx = [6, 15]
    elif file_type == 'QPE':
        np_data = np.array(ds['Precipitation'][:]).astype(np.float32)
        np_data = np_data[np.newaxis, :]
        in_out_idx = [0, 1]
    else:
        np_data = None
        in_out_idx = None
    ds.close()
    r = reference_cache.get_key_in_cache(file_info['position'])
    if r is None:
        print(f'get reference of {file_info["position"]}')
        r = reference_cache.add_key(file_info['position'], get_reference(ld=file_info['position']))

    np_data = getNPfromHDF_worker(np_data, file_info['start'], r=r, ld=file_info['position'], to_file=False,
                                  in_out_idx=in_out_idx)
    return np_data


def getNPfromHDF_worker(read_np_data, current_date, data_dir=None, ld=None, r=None, to_file=True, in_out_idx=(6, 15)):
    tmp_dir = 'make_temp' + str(randint(10000000, 99999999))
    file = current_date.strftime("%H%M") + '.npy'
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = f'{current_date.year}{current_date.month}{current_date.day}{file.replace(".npy", "")}'
    np2tif(read_np_data, tmp_dir, save_path, print_log=False)
    in_idx, out_idx = in_out_idx
    np_data = np.zeros((out_idx - in_idx, x_range['height'], x_range['width']), dtype=np.float16)
    for i in range(in_idx, out_idx):
        out_path = f'{tmp_dir}/{save_path}-{i}-ctrl.tif'
        registration(f'{tmp_dir}/{save_path}-{i}.tif', out_path, ld, *r)
        img = cv2.imread(out_path, -1)
        if np.isnan(img).any():
            shutil.rmtree(tmp_dir)
            return None
        np_data[i - in_idx] = img.astype(np.float16)
    # raise Exception('manual stop')
    shutil.rmtree(tmp_dir)
    if to_file:
        os.makedirs(f'{data_dir}/dataset/{current_date.year}/{current_date.month}/{current_date.day}', exist_ok=True)
        np.save(
            f'{data_dir}/dataset/{current_date.year}/{current_date.month}/{current_date.day}/{convert_file2idx(file)}',
            np_data)
    else:
        return np_data


def prase_filename(filename):
    m_list = filename.replace('.HDF', '').split('_')
    return {'satellite': m_list[0], 'sensor': m_list[1], 'area': m_list[3],
            'position': int(m_list[4][:3]), 'start': datetime.strptime(m_list[9], '%Y%m%d%H%M%S'),
            'end': datetime.strptime(m_list[10], '%Y%m%d%H%M%S'), 'resolution': m_list[11]}


def registration(input_path, out_path, ld, spatial_reference, gcps_list):
    """
    基于python GDAL配准
    :param input_path: 需要配准的栅格文件
    :param out_path: 输出配准后的栅格文件位置
    :param top_left: 左上角坐标
    :param bottom_right: 右下角坐标
    :param ik: 行空白分辨率
    :param jk: 列空白分辨率
    :return:
    """
    # 打开栅格文件
    dataset = gdal.Open(input_path, gdal.GA_Update)
    # 添加控制点
    dataset.SetGCPs(gcps_list, spatial_reference.ExportToWkt())
    # tps校正 重采样:最邻近法
    gdal.Warp(out_path, dataset,
              format='GTiff',
              outputBounds=[ld + x_range['left'], x_range['bottom'], ld + x_range['right'], x_range['top']],
              resampleAlg=gdal.GRIORA_NearestNeighbour,
              width=x_range['width'],
              height=x_range['height'],
              tps=True,
              dstSRS='EPSG:4326')


def calculate_solar_zenith_angle(latitude, longitude, elevation, date_time):
    # 创建观测者对象
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.elev = elevation

    # 设置观测时间
    observer.date = date_time

    # 创建太阳对象
    sun = ephem.Sun()

    # 计算太阳天顶角
    sun.compute(observer)
    solar_alt_angle = math.degrees(sun.alt)  # 高度角 sin(alt)
    solar_azimuth = math.degrees(sun.az)  # 方位角 sin(az-90)
    date = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') + timedelta(hours=longitude / 15)
    if 12 < date.hour < 24:
        solar_alt_angle = 180 - solar_alt_angle
    return solar_azimuth, solar_alt_angle


def make_alt(coord, now_date):
    alt = np.zeros((coord.h, coord.w))
    for i in range(coord.h):
        for j in range(coord.w):
            lat, lon = coord.top - i * coord.y_res, coord.left + j * coord.x_res
            alt[i, j] = calculate_solar_zenith_angle(lat, lon, 0, now_date)[1]
    # alt = (np.cos(alt / 180 * np.pi) + 0) / 1
    alt = (np.cos(alt / 180 * np.pi) + 0.8827106106270867) / 0.11592633264618261
    return alt


# 设置经纬度范围,限定为中国
# 注意指定crs关键字,否则范围不一定完全准确
extents = [100, 140, 20, 60]
proj = ccrs.PlateCarree()


def _get_color_normalization(data, colors):
    max_value = colors[-1][0]
    min_value = colors[0][0]
    data[data < min_value] = min_value
    data[data > max_value] = max_value
    data = (data - min_value) / (max_value - min_value)
    new_colors = []
    for color in colors:
        new_colors.append([(color[0] - min_value) / (max_value - min_value), color[1]])
    return data, new_colors


def make_fig(file_name, root_path, out_folder=None, tz='UTC',
             colors=((0, '#1E90FF'), (0.1, '#1874CD'), (0.2, '#3A5FCD'), (0.3, '#0000CD'), (1, '#9400D3')),
             area=((100, 140, 10), (20, 60, 10)), font_size=20, corp=(0, 0, None, None),
             zoom_rectangle=(310 * 5, 300 * 5, 50 * 5, 40 * 5), docker=(300, 730), dpi=500, filter=0.3, exposure=None):
    # corp = [92, 31, 542, 456]
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extents, crs=proj)
    # 读取TIFF数据
    elevation = None
    if type(file_name) == list:
        for each_file in file_name:
            file_path = os.path.join(root_path, each_file)
            with rasterio.open(file_path) as dataset:
                el_rd = dataset.read(1)
                elevation[elevation < filter] = np.nan
                if elevation is None:
                    elevation = el_rd
                else:
                    elevation += el_rd
    else:
        file_path = os.path.join(root_path, file_name)
        with rasterio.open(file_path) as dataset:
            elevation = dataset.read(1)
        elevation[elevation < filter] = np.nan
    elevation, colors = _get_color_normalization(elevation, colors)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    # 添加各种特征
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none',
                                        linewidth=0.4)
    ax.add_feature(land)
    ax.imshow(elevation, origin='upper', extent=extents, transform=proj, cmap=cmap)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)
    # ax.add_feature(cfeature.BORDERS)
    # 添加网格线
    # ax.gridlines(linestyle='--')
    # 设置大刻度和小刻度
    tick_proj = ccrs.PlateCarree()
    ax.set_xticks(np.arange(area[0][0], area[0][1] + 1, area[0][2]), crs=tick_proj)
    # ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=tick_proj)
    ax.set_yticks(np.arange(area[1][0], area[1][1] + 1, area[1][2]), crs=tick_proj)
    # ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=tick_proj)
    # 利用Formatter格式化刻度标签
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    if out_folder is None:
        if type(file_name) == list:
            file_name = file_name[0]
            file_dir = 'concate'
        else:
            file_dir = 'outs'
    else:
        file_dir = out_folder
    os.makedirs(os.path.join(root_path, file_dir), exist_ok=True)
    file_name = file_name.replace('.tif', '.png')
    file_title = datetime.strptime(file_name.split('-')[0].replace('target_', ''), '%Y%m%d_%H%M%S')
    file_name = file_title.strftime('%Y%m%d_%H%M%S.png')
    exposure = exposure if exposure else (60 if file_dir == 'concate' else 15)
    if tz == 'BJT':
        exposure_end = (file_title + timedelta(hours=8) + timedelta(minutes=exposure)).strftime('%H:%M')
        file_title = (file_title + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
    else:
        exposure_end = (file_title + timedelta(minutes=exposure)).strftime('%H:%M')
        file_title = file_title.strftime('%Y-%m-%d %H:%M')
    ax.set_title(file_title + f'-' + exposure_end + f' ({tz})', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.title(fontsize=font_size)
    plt.savefig(os.path.join(root_path, file_dir, file_name))
    if zoom_rectangle is not None:
        read_png = cv2.imread(os.path.join(root_path, file_dir, file_name))
        read_png = zoomAndDock(read_png, zoom_rectangle, docker, scale_factor=5, border=14)
        cv2.imwrite(os.path.join(root_path, file_dir, file_name), read_png)
    crop_png(os.path.join(root_path, file_dir, file_name), left=corp[0], top=corp[1], right=corp[2], bottom=corp[3])
    # plt.show()


if __name__ == '__main__':
    # color = ((0, '#D6D5B7'), (0.4, '#8CC7B5'), (0.6, '#F0EEE1'), (0.8, '#FEE7E9'), (1, '#FFFFFF'))
    color = ((0, '#D6D5B7'), (3, '#8CC7B5'), (5, '#F0EEE1'), (6, '#FEE7E9'), (8, '#FFFFFF'))
    make_fig('20240808_000000-qpe.tif', r'C:\Users\ECNU\Desktop\240808\prec\x1step\image', out_folder='test',
             colors=color, zoom_rectangle=[310 * 5, 300 * 5, 50 * 5, 40 * 5], docker=[300, 730], dpi=500)
