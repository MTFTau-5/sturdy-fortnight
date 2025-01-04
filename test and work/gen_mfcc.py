import os
import glob
import psutil
import pickle
from tqdm import tqdm
import audioflux as af
from joblib import Parallel , delayed

def mfcc_feature(file_name):
    audio, sr = af.read(file_name)
    mfcc, _ = af.mfcc(audio, samplate=sr)
    return mfcc

def gen_data(snr):
    data = []
    
    # 定义device目录列表等
    devices = ['fan', 'pump', 'slider', 'valve']
    ids = ['id_00', 'id_02', 'id_04', 'id_06']
    labels = ['abnormal','normal']

    # 定义映射字典
    devicem = {'fan': 0, 'pump': 1, 'slider': 2, 'valve': 3}
    labelm = {'abnormal': 1, 'normal': 0}
    
    for device in devices:
        print("processing {} {}".format(snr, device))
        for did in ids:
            for label in labels:
                wav_files = [os.path.basename(f) for f in glob.glob(os.path.join(snr, device, did, label, "**.wav"), recursive=True)]
                mfccs = Parallel(n_jobs=psutil.cpu_count(logical=True), verbose=0)(delayed(lambda i: mfcc_feature(os.path.join(snr, device, did, label, i)))(i) for i in tqdm(wav_files))

                for mfcc in mfccs:
                    data.append([mfcc, devicem[device], labelm[label]])
    return data


if __name__ == '__main__':

    # 进入data目录
    os.chdir('/home/mtftau-5/workplace/dataset')

    data = []
    
    # 定义snr目录列表
    snrs = ['-6dB', '0dB', '6dB']

    # 定义映射字典
    snrm = {'-6dB': -1, '0dB': 0, '6dB': 1}
    
    for snr in snrs:
        data.append(gen_data(snr))

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)