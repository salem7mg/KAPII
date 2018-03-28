"""
�T�v
    KPI�f�[�^�ُ̈픻��&�O���t�o�͏������s���B


��ʐ���

    ���C��
    txBox1�{openb1      :text+button     :���͂j�o�h�t�H���_�[�{�I��
    txBox2�{openb2      :text+button     :�o�̓O���t�t�H���_�[�{�I��
    rbtn2               :radio           :HUA or ZTE �I��
    nwb                 :radio           :�O���t�ڍבI��
    rbtn4�@�@�@�@�@�@�@ :radio           :�������[�h�I��  u'���f�����O�{�O���t', u'���f�����O', u'�O���t'
    rbtn�@�@�@�@�@�@�@�@:radio           :��͊��ԑI��    u'3 Days', u'1 Week', u'Free'
        s_pane          :text            :�J�n��
        e_pane          :text            :�I����
    stbtn               :button          :�X�^�[�g�{�^��
    pb1                 :progress        :�v���O���X�o�[

"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import HOURLY, DateFormatter, drange, HourLocator, DayLocator
from matplotlib.ticker import ScalarFormatter

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.tix as tix
import tkinter.filedialog as tkfd
from tkinter import messagebox

from aux_tkinter import DatePane, LabelPane, Menubar, NewWindowButton, TabView, ImageLabel, ImagePane, TreeView, \
    Dialogbox, Meter, Slider, DebugPane
from aux_tkinter import ThreadedGUI, ThreadingException
from aux_misc import Filemngr, Debugger
from aux_plot import Mpltfont
from aux_math import Arr, Spectrum, Time
from aux_pool import Pooler
from aux_csv import CSV

CSV.ENCODING = "cp932"

import pandas as pd
import pandas.tseries.offsets as offsets
import numpy as np
import configparser as ConfigParser
import _thread
import datetime
import time
import math
import random
import os
import sys
import csv
import glob
import re
import shutil
import logging
import gc
import traceback
import multiprocessing as mp
import labeler as la


# log_fmt = '%(asctime)s- %(name)s - %(levelname)s - %(message)s'
# logging.basicConfig(filename=os.getcwd()+'/example.log', format=log_fmt)

def todatetime(str):
    """
    ���t�����^�C�v�֕ϊ�
    :param str:string�F�@yyyy/mm/dd hh:mm:ss or yyyy/mm/dd hh:mm
    :return:DateTime
    """
    str = str.replace("/", "-")
    try:
        t = datetime.datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        t = datetime.datetime.strptime(str, "%Y-%m-%d %H:%M")
    return t


# --------------------const globals--------------------

NOT_KPI = ["DateTime", "Datetime", "BAND", "COUNT", "HW"]

# ����������
day = 24
week = 7
hrs_in_week = 24 * 7

# GUI�F
C_GOLD = "#EECD59"
C_BLUE = "#57A4C8"

# �p�X�ݒ�
global parentdir
global maindir
global i1r
global i2r
global i3r
global i4r

parentdir, maindir = Filemngr.getdir("main")
i1r = Filemngr.resource_path("i1.gif")
i2r = Filemngr.resource_path("i2.gif")
i3r = Filemngr.resource_path("i3.gif")
i4r = Filemngr.resource_path("i4.gif")

# --------------------globals--------------------

tpara = 1.05  # 臒l�̕␳�l
epochs = 100  # �G�|�b�N��
procs = 3  # �v���Z�X��
tempfiles = set()  # temp�t�@�C���̃��X�g
global gui  # threaded gui

# ini�t�@�C���֘A
config2t = {}
TabSort = []

defined_main_keras = False


def keras_import():
    """
    keras_import ����с@autoencoder�p�����[�^�ݒ�
    :return:
    """
    global tf, backend, Input, Dense, Model, autoencoder
    global tf_session, tf_graph
    import tensorflow as tf
    from keras import backend
    from keras.layers import Input, Dense
    from keras.models import Model
    tf_session = backend.get_session()
    tf_graph = tf.get_default_graph()
    encoding_dim = 21
    input_data = Input(shape=(24,))
    encoded = Dense(encoding_dim, activation='relu')(input_data)
    decoded = Dense(24, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_data, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# --------------------process functions--------------------

def initializer(l):
    """
    initialize mutex lock

    :param l:
    :return:
    """
    # signal.signal(signal.SIGINT, signal.SIG_IGN)  #for ctrl-c
    global lock
    lock = l


def write_TMAX(path, str):
    """
    KPI��MAX�l�t�@�C���o��
    :param path     :string
    :param str      :string
    :return:
    """
    with open(path, 'w') as f:
        f.write(str)


def read_TMAX(path):
    """
    KPI��MAX�l�ǂݍ���
    :param path:string
    :return:
    """
    r = None
    with open(path, 'r') as f:
        r = float(f.read())
    return r


def isconst(x):
    """
    ���X�g���S�ē��������̏ꍇtrue
    :param x    :np.array
    :return:bool
    """
    x = np.array(x)
    x_notnan = x[~np.isnan(x)]
    if len(x) == 0:
        return True, None
    elif len(x_notnan) == 0:
        return True, np.nan
    elif len(x_notnan) == 1:
        return True, x_notnan[0]
    else:
        x_const = x_notnan[x_notnan == x_notnan[0]]
        if len(x_const) == len(x_notnan):
            return True, x_notnan[0]
        else:
            return False, None


def process_kpis(gui_arg, dftrain, dfkpi, init_modelpath, modelpath, flag_model, flag_graph, ixs, ixe, title, outpath,
                 ijyoufile, tpara, epochs, config2t, sd):
    """
    �v���Z�X���C������
    :param gui_arg:
    :param dftrain:pd               �g���[�j���O
    :param dfkpi:pd                 �j�o�h�f�[�^
    :param init_modelpath:string    ���f���i�[Path+�t�@�C����
    :param modelpath:string         ���f���i�[Path+�t�@�C����
    :param flag_model:bool          ���f���쐬�L��
    :param flag_graph:bool          �O���t�쐬�L��
    :param ixs:int                  �J�n�N�����̈ʒu
    :param ixe:int                  �I���N�����̈ʒu
    :param title:string             KPI�^�C�g��
    :param outpath:string           �O���t�i�[Path
    :param ijyoufile:string         �ُ�t�@�C���i�[Path+�t�@�C����
    :param tpara:init               臒l�̕␳�l
    :param epochs:init              epoch��
    :param config2t:dict            �ݒ�t�@�C��
    :param sd:dict                  �������[�N
    :return:pd                      ����ُ탉�x��
    """
    if len(dftrain.columns) == 1:
        return None
    if not "lock" in globals():
        global autoencoder
    else:
        import tensorflow as tf
        from keras import backend
        from keras.layers import Input, Dense
        from keras.models import Model
        tf_session = backend.get_session()
        tf_graph = tf.get_default_graph()
        encoding_dim = 21
        input_data = Input(shape=(24,))
        encoded = Dense(encoding_dim, activation='relu')(input_data)
        decoded = Dense(24, activation='sigmoid')(encoded)
        autoencoder = Model(input=input_data, output=decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    kpis = dftrain.columns.tolist()[1:]
    d1 = dfkpi['DateTime'].iloc[len(dfkpi['DateTime']) - ixs].strftime("%Y%m%d")
    d2 = dfkpi['DateTime'].iloc[len(dfkpi['DateTime']) - ixe - 1].strftime("%Y%m%d%H")
    d12 = d1 + "-" + d2
    if sd["exit"]:
        return None
    for kpi in kpis:
        if kpi.endswith("_label"):
            break
        try:
            p1 = "D" + (config2t[kpi]["Tab01"]["Tab01"].split("-"))[0]  # Dxx
            p2 = Filemngr.tofilestr(config2t[kpi]["Grp01"]["Grp01"])  # name
            p3 = Filemngr.tofilestr(kpi)  # name (short)
            imgpath = outpath + "/" + "_".join([p1, p2, p3, d12])
            modelfl = modelpath + kpi.replace('/', '_').replace(' ', '_').replace('>', '_').replace('<', '_')
            idx_nan = np.where(np.isnan(dfkpi[kpi]))[0].tolist()
            dftrain[kpi].fillna(0, inplace=True)
            dfkpi[kpi].fillna(0, inplace=True)
            flag_ijyou = False
            flag_all_zero = False
            flag_ae_load = False
            flag_const, cval = isconst(dftrain[kpi])
            if flag_const and cval == 0:
                flag_all_zero = True
            if flag_all_zero:
                if flag_graph:
                    for i in range(0, len(dfkpi)):
                        if dfkpi[kpi][i] == 0:
                            dfkpi.set_value(i, kpi + "_label", 0)
                        else:
                            dfkpi.set_value(i, kpi + "_label", 1)
                            flag_ijyou = True
                if "lock" in globals():
                    lock.acquire()
                    sd["counter"] = sd["counter"] + 2
                    lock.release()
                else:
                    sd["counter"] = sd["counter"] + 2
                    gui_arg.updatepb()
                if sd["exit"]:
                    return None
            else:
                # �f�[�^����
                start = time.time()
                idx_train, idx_even = get_idx_train(kpi, dftrain, dfkpi, idx_nan, imgpath, sd)
                TMAX = max(np.nanmax(dftrain[kpi]), np.nanmax(dfkpi[kpi]))  # �ő�l�Z�o
                TMIN = min(np.nanmin(dftrain[kpi]), np.nanmin(dfkpi[kpi]))
                if abs(TMIN) > abs(TMAX):
                    TMAX = TMIN
                flag_train = not (np.isnan(TMAX) or (TMAX == 0) or (idx_train is None))
                end = time.time()
                print(kpi, u"�f�[�^����", end - start)

                if "lock" in globals():
                    lock.acquire()
                    sd["counter"] = sd["counter"] + 1
                    lock.release()
                else:
                    sd["counter"] = sd["counter"] + 1
                    gui_arg.updatepb()
                if sd["exit"]:
                    return None

                # ���f���쐬�E���[�h
                start = time.time()
                if flag_train:
                    flag_model_exists = (os.path.exists(modelfl + '.h5') and os.path.exists(modelfl + '.tmax'))
                    train_data = np.array([np.array(dfkpi[kpi][i:i + 24]) for i in idx_even])
                    # ���f���쐬
                    if flag_model or (not flag_model_exists):
                        """
                        for layer in autoencoder.layers:
                            if hasattr(layer, 'kernel_initializer'):
                                layer.kernel.initializer.run(session=tf_session)
                        """
                        autoencoder.load_weights(init_modelpath)
                        autoencoder.fit(train_data / TMAX, train_data / TMAX, epochs=epochs, batch_size=128,
                                        shuffle=True, verbose=0, validation_data=(train_data / TMAX, train_data / TMAX))
                        write_TMAX(modelfl + '.tmax', str(TMAX))
                        autoencoder.save_weights(modelfl + '.h5')
                        flag_ae_load = True
                    flag_model_exists = (os.path.exists(modelfl + '.h5') and os.path.exists(modelfl + '.tmax'))
                    if flag_graph:
                        # ���f�����[�h
                        if (not flag_ae_load) and flag_model_exists:
                            rTMAX = read_TMAX(modelfl + '.tmax')
                            if abs(rTMAX) >= abs(TMAX):
                                TMAX = rTMAX
                                autoencoder.load_weights(modelfl + '.h5')
                                flag_ae_load = True
                            else:  # TMAX > rTMAX
                                """
                                for layer in autoencoder.layers:
                                    if hasattr(layer, 'kernel_initializer'):
                                        layer.kernel.initializer.run(session=tf_session)
                                """
                                autoencoder.load_weights(init_modelpath)
                                autoencoder.fit(train_data / TMAX, train_data / TMAX, epochs=epochs, batch_size=128,
                                                shuffle=True, verbose=0,
                                                validation_data=(train_data / TMAX, train_data / TMAX))
                                write_TMAX(modelfl + '.tmax', str(TMAX))
                                autoencoder.save_weights(modelfl + '.h5')
                                flag_ae_load = True
                        # 臒l
                        if flag_ae_load:
                            ans = get_ans(kpi, dftrain, dfkpi, idx_train, autoencoder, modelfl, TMAX, tpara, config2t,
                                          imgpath, sd)
                            m = 0
                            for k in range(len(dfkpi) - len(ans), len(dfkpi)):
                                if ans[m] == 1:
                                    flag_ijyou = True
                                dfkpi.set_value(k, kpi + "_label", ans[m])
                                m = m + 1
                end = time.time()
                print(kpi, u"���f���쐬�E���[�h", end - start)

                if "lock" in globals():
                    lock.acquire()
                    sd["counter"] = sd["counter"] + 1
                    lock.release()
                else:
                    sd["counter"] = sd["counter"] + 1
                    gui_arg.updatepb()
                if sd["exit"]:
                    return None
            # �O���t
            start = time.time()
            if flag_graph:
                dfkpi.ix[idx_nan, kpi] = np.nan
                graph5(kpi, dfkpi, ixs, ixe, title + kpi, flag_ijyou, flag_ae_load, flag_all_zero, imgpath, ijyoufile,
                       config2t, sd)
            end = time.time()
            print(kpi, u"臒l�E�O���t", end - start)

            if "lock" in globals():
                lock.acquire()
                sd["counter"] = sd["counter"] + 1
                lock.release()
            else:
                sd["counter"] = sd["counter"] + 1
                gui_arg.updatepb()
            if sd["exit"]:
                return None

        except Exception as e:
            e.args += (kpi,)
            raise
    lblcols = []
    for colname in dfkpi.columns.tolist():
        if colname.endswith("_label"):
            lblcols.append(colname)
    return dfkpi.loc[:, lblcols]


def get_idx_train(kpi, dftrain, dfkpi, idx_nan, imgpath, sd):
    """
	�g���[�j���O����ʒu�擾
    :param kpi:string               �j�o�h��
    :param dftrain:pd               �g���[�j���O�f�[�^
    :param dfkpi:pd                 �j�o�h�f�[�^
    :param idx_nan:np               �j�o�h�f�[�^��nan�̈ʒu
    :param imgpath:string�@�@�@�@�@ �O���t�o��Path+�t�@�C����
    :param sd:dict                  �������[�N
    :return:idx_train, idx_even     24*7����f�[�^�X�^�[�g�C���f�b�N�X�A�����C���f�b�N�X
    """
    idx9 = dftrain.index[dftrain[kpi + '_label'] == 9]
    if len(idx9) != 0:
        z, r = Arr.slice_zero(dftrain[kpi])
        if len(z) + len(r) <= hrs_in_week * 2:
            dfkpi.ix[idx9, kpi + '_label'] = 1
        elif len(r) == 0:
            dfkpi[kpi + '_label'].loc[:len(dftrain) - 1] = 0
        elif len(r) <= hrs_in_week * 2:
            dfkpi[kpi + '_label'].loc[:len(z) - 1] = 1
            dfkpi[kpi + '_label'].loc[len(z):len(dftrain) - 1] = 9
        else:
            start = time.time()
            has_24h_period = False
            if sd["debug"]:
                has_24h_period = Spectrum.analyze(dftrain[kpi].as_matrix().tolist(), saveplot=imgpath + "_spectrum.png")
            else:
                has_24h_period = Spectrum.analyze(dftrain[kpi].as_matrix().tolist())
            end = time.time()
            print(kpi, "spectrum", end - start)
            start = time.time()
            if has_24h_period:
                t = dftrain["DateTime"].tolist()
                y = dftrain[kpi].tolist()
                dfkpi[kpi + '_label'].loc[0:len(dftrain) - 1] = la.labelit(t, y, title="", savepath=None, plot=False)
                dfkpi.ix[idx_nan, kpi + '_label'] = 1
            end = time.time()
            print(kpi, "autolabel", end - start)
    p = dfkpi[kpi + "_label"].tolist()[0:len(dftrain) - 1]
    #Arr.partition_span�̓X�^�[�g�C���f�b�N�X��Ԃ�,idx_even�͕����̃C���f�b�N�X
    #�S�āi168�j�����Ă��Ȃ��ꍇnan��Ԃ�
    idx_train, idx_even, span = Arr.partition_span(p, 0, 24, hrs_in_week)
    if span != hrs_in_week:
        idx_train = None
        idx_even = None
    return idx_train, idx_even


def get_ans(kpi, dftrain, dfkpi, idx_train, autoencoder, modelfl, TMAX, tpara, config2t, imgpath, sd):
    """
    �g���[�j���O�@����с@����ُ픻��
    :param kpi:string                           �j�o�h��
    :param dftrain:pd                           �g���[�j���O�f�[�^
    :param dfkpi:pd                             �j�o�h�f�[�^
    :param idx_train:partition start indices    24*7����f�[�^�X�^�[�g�C���f�b�N�X
    :param autoencoder:autoencoder              autoencoder�I�u�W�F�N�g
    :param modelfl:string                       ���f���t�@�C����
    :param TMAX:float                           �j�o�h�ő�l
    :param tpara:int                            臒l�␳
    :param config2t:dict                        �ݒ�t�@�C��
    :param imgpath:string                       �o�̓O���tPath + �O���t��
    :param sd:dict                              �������[�N
    :return ans:numpy                           ����ُ�T�C��
    """
    train_data0 = Arr.partition_split(dfkpi[kpi], 24, len(dftrain) - 1)
    test_data0 = np.array([np.array(dfkpi[kpi][i:i + 24]) for i in range(len(dftrain) - 24, len(dfkpi[kpi]) - 24)])
    x_train_out = autoencoder.predict(train_data0 / TMAX) * TMAX
    # ���l�[����l�̐�Βl�Z�o
    diff_train = []
    for i in range(len(x_train_out)):
        diff_train.append(abs(x_train_out[i][-1] - train_data0[i][-1]))
    diff_train = [diff_train[i] for i in idx_train]
    diff_train = np.array(diff_train)
    # ��Βl���ړ����ρ�臒l�Z�o
    div0 = pd.ewma(diff_train, span=hrs_in_week)
    div1 = np.nanmax(div0[hrs_in_week:len(div0)])
    x_test_out = autoencoder.predict(test_data0 / TMAX) * TMAX
    # autoencod-real�����Z�o
    diff_test = []
    for i in range(len(x_test_out)):
        diff_test.append(abs(x_test_out[i][-1] - test_data0[i][-1]))
    diff_test = np.array(diff_test)
    div2 = pd.ewma(np.append(diff_train, diff_test), span=hrs_in_week)
    div2 = div2[len(div2) - len(diff_test):len(div2)]
    ans = np.ones(len(test_data0))
    for i in range(len(test_data0)):
        k = test_data0[i][0]
        if div2[i] <= div1 * tpara:
            if int(config2t[kpi]['CriCfg']['CriCfg']) == 0:
                ans[i] = 0
            elif int(config2t[kpi]['CriCfg']['CriCfg']) == 1:
                if k > float(config2t[kpi]['CriVal']['CriVal']):
                    ans[i] = 0
            else:
                if k < float(config2t[kpi]['CriVal']['CriVal']):
                    ans[i] = 0
    if sd["debug"]:
        shift = max(np.nanmax(diff_train), np.nanmax(diff_test)) * 1.5
        diff_train_a = []
        diff_train_b = []
        for i in range(len(x_train_out)):
            if i in idx_train:
                diff_train_a.append(train_data0[i][0] + shift)
                diff_train_b.append(np.nan)
            else:
                diff_train_a.append(np.nan)
                diff_train_b.append(train_data0[i][0] + shift)
        diff_test_a = []
        for i in range(len(x_test_out)):
            diff_test_a.append(test_data0[i][0] + shift)
        fig, ax = plt.subplots(figsize=(19, 9))
        n_train = len(diff_train)
        n_test = len(diff_test)
        n_pad = len(diff_train_a) - n_train
        n = n_pad + n_train + n_test
        pad = [np.nan] * n_pad
        ax.plot(np.arange(n), diff_train_a + diff_test_a, label="actual_used", color="green")
        ax.plot(np.arange(n), diff_train_b + [np.nan] * n_test, label="actual_notused", color="red")
        ax.plot(np.arange(n), pad + diff_train.tolist() + [np.nan] * n_test, label="diff_train")
        ax.plot(np.arange(n), pad + [np.nan] * n_train + diff_test.tolist(), label="diff_test")
        ax.plot(np.arange(n), [div1 * tpara] * n, label="div1")
        ax.plot(np.arange(n), pad + div0.tolist() + div2.tolist(), label="div")
        ax.legend()
        plt.savefig(imgpath + "_ae.png")
        fig.clf()
        plt.close()
    return ans


def graph5(kpi, dfkpi, ixs, ixe, title, flag_ijyou, flag_ae_load, flag_all_zero, imgpath, ijyoufile, config2t, sd):
    """
    �O���t�쐬
    :param kpi:string                       �j�o�h��
    :param dfkpi:df                         �j�o�h�f�[�^
    :param ixs:int                          �O���t�J�n�ʒu
    :param ixe:int                          �O���t�I���ʒu
    :param title:string                     �O���t�^�C�g��
    :param flag_ijyou:bool                  �ُ�L��
    :param flag_ae_load:bool                ���f�����[�h�L��
    :param flag_all_zero:bool               �g���[�j���O�f�[�^ALLzero
    :param imgpath:string                   �O���t�o��Path+�O���t��
    :param ijyoufile:string                 �O���t�o��Path+�ُ�t�@�C����
    :param config2t:dict                    �ݒ�t�@�C��
    :param sd:dict                          �������[�N
    :return:
    """
    Mpltfont.set('IPAexGothic')
    n = len(dfkpi[kpi])
    x = dfkpi['DateTime'][n - ixs:n - ixe].tolist()
    y = dfkpi[kpi][n - ixs:n - ixe].tolist()
    z = dfkpi[kpi + "_label"][n - ixs:n - ixe].tolist()
    z = [1 if z[i] == 1 else 0 for i in range(len(z))]
    # ��������i�����l�A�����A�ő�l�j
    # �ő�A�ŏ��l�Z�o
    lo = np.nanmin(y)
    hi = np.nanmax(y)
    ymid = (hi + lo) / 2
    ymin_1 = 0
    ymax_1 = 0
    ystep_12 = 0
    # �ő�l���P�D�T�{�ɂ��ő�l�␳
    if hi >= lo and hi != 0:
        ymin_1 = lo
        ystep_1 = hi
        if hi > 0:
            ystep_1 = 10 ** (int(math.log10(hi * 1.5)))
        else:
            ystep_1 = 2
        ymax_1 = ((int)(hi * 1.5 / ystep_1) + 1) * ystep_1
    # �ݒ�t�@�C���̒l���f
    if config2t[kpi]['MaximumScale Auto']['MaximumScale Auto'] == ' False':
        ymax_1 = float(config2t[kpi]['MaximumScale']['MaximumScale'])
    if config2t[kpi]['MinimumScale Auto']['MinimumScale Auto'] == ' False':
        ymin_1 = float(config2t[kpi]['MinimumScale']['MinimumScale'])
    if config2t[kpi]['MajorUnit Auto']['MajorUnit Auto'] == ' False':
        ystep_12 = float(config2t[kpi]['MajorUnit']['MajorUnit'])
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.title(title, fontsize=20)
    plt.ylabel(config2t[kpi]['AxisTitle']['AxisTitle'])
    if ymin_1 <= ymax_1 and ystep_12 != 0:
        ax.set_ylim((ymin_1, ymax_1))
        y2 = np.arange(ymin_1, ymax_1, ystep_12)
        ax.set_yticks(y2)
        ax.set_yticklabels(y2, fontsize=15, weight='bold')
    if len(x) <= 24 * 3:
        ax.xaxis.set_major_locator(HourLocator(interval=1))
    else:
        ax.xaxis.set_major_locator(DayLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%y-%m-%d %H:%M'))
    ax.set_xlim([x[0], x[-1]])
    ax.xaxis.grid()
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 4))
    ax.yaxis.offsetText.set_fontsize(22)
    if flag_ijyou and (not np.isnan(ymid)):
        labeledt(ax, x, z, ymid, "r")
    p1, = ax.plot(x, y, label=kpi)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(22)
    lgd = ax.legend([p1], [kpi], bbox_to_anchor=(0.5, -0.26), loc='center', borderaxespad=0, ncol=1, fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=14, weight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(imgpath + ".png", transparent=False, bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.clf()
    plt.close()
    # �ُ�ꗗ�t�@�C����������
    if (not flag_ijyou) and flag_all_zero:
        return
    if (not flag_ijyou) and flag_ae_load:
        return
    if flag_ijyou:
        ijyou = 0
        ijyou_st = None
        ijyou_ed = None
        for i in range(n - ixs, n - ixe, 1):
            if dfkpi[kpi + "_label"][i] == 1:
                ijyou_st = dfkpi['DateTime'][i].strftime("%Y-%m-%d %H:00")
                ijyou = 1
                break
        if ijyou == 1:
            for i in range(n - ixe - 1, n - ixs - 1, -1):
                if dfkpi[kpi + "_label"][i] == 1:
                    ijyou_ed = dfkpi['DateTime'][i].strftime("%Y-%m-%d %H:00")
                    break
            f = open(ijyoufile, 'a')
            f.write(kpi + "," + ijyou_st + "," + ijyou_ed + ",\n")
            f.close()
    elif (not flag_all_zero) and (not flag_ae_load):
        f = open(ijyoufile, 'a')
        f.write(kpi + ",,,������\n")
        f.close()
        return


def labeledt(ax, dt, dtlabel, y, c):
    """
    �O���t��ُ̈�T�C���Z�b�g
    :param ax:subplot               �T�u�v���b�g
    :param dt:                      ���t��
    :param dtlabel:                 �ُ�L��
    :param y:list                   �`��c�ʒu
    :param c:list                   �F
    :return:
    """
    for k in range(len(dt)):
        if k > 0 and k < len(dt) - 1:
            if dtlabel[k - 1] == 0 and dtlabel[k] == 1 and dtlabel[k + 1] == 0:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 0 and dtlabel[k] == 1 and dtlabel[k + 1] == 1:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 1 and dtlabel[k] == 1 and dtlabel[k + 1] == 0:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 1 and dtlabel[k] == 1 and dtlabel[k + 1] == 1:
                ax.text(dt[k], y, u"�\", color=c, horizontalalignment="center")
        elif k == 0:
            if dtlabel[k] == 1 and dtlabel[k + 1] == 0:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")
            elif dtlabel[k] == 1 and dtlabel[k + 1] == 1:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")
        elif k == len(dt) - 1:
            if dtlabel[k - 1] == 1 and dtlabel[k] == 1:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 0 and dtlabel[k] == 1:
                ax.text(dt[k], y, u"��", color=c, horizontalalignment="center")


# --------------------read ini file--------------------

def show_config(ini):
    """
    �ݒ�t�@�C���̑S�Ă̓��e��\������

    :param ini:object
    :return:dict
    """
    config = {}
    for section in ini.sections():
        if section == "Vender":
            continue
        # config.append([section,show_section(ini, section)])
        config[section] = show_section(ini, section)
    return config


def show_section(ini, section):
    """
    �ݒ�t�@�C���̓���̃Z�N�V�����̓��e��\������

    :param ini:object
    :param section:string
    :return:string
    """
    Key = {}
    for key in ini.options(section):
        Key[key] = show_key(ini, section, key)
    # print(Key)
    return Key


def show_key(ini, section, key):
    """
    �ݒ�t�@�C���̓���Z�N�V�����̓���̃L�[���ځi�v���p�e�B�j�̓��e��\������

    :param ini:object
    :param section:string
    :param key:string
    :return:dict
    """

    """   
    if ini.get(section, key).find("\n") != -1:
        l = ini.get(section, key).split("\n")
        l2={}
        #print(l)
        for i in range(1,len(l)):
            s = l[i]

            s2=s.split(":")
            if len(s2)>1:
                l2[s2[0]] = s2[1]
            else:
                l2 = s2[0]
            #print(l2)
        #l2={key:l2}    
    else:
    """
    l2 = {key: ini.get(section, key)}
    # l2={ini.get(section, key)}
    return l2


# --------------------gui--------------------

def createnewwindow(nwb, parent):
    """
    �T�u��ʁFini�t�@�C���I��������ʂ̕\��
    :param nwb:childObject
    :param parent:objct
    :return:
    """
    mbar = Menubar(nwb.root, nwb.on_close, parent.inifilename, menu_load, menu_save, dir_load=parent.default_inipath,
                   dir_save=parent.default_inipath, arg=parent)


def get_tabs(menu, filename):
    """
    �T�u��ʁF�ݒ���Ǎ��݂���уO���[�v�\�[�g
    :param menu:object
    :param filename:string
    :return:list,list
    """
    master = menu.master  # current window
    parent = menu.arg  # parent window
    cp = ConfigParser.SafeConfigParser(strict=False)
    cp.optionxform = str
    cp.read(filename)
    if not cp.has_section("Vender"):
        messagebox.showerror(title=u"ini�t�@�C���ǂݍ��݃G���[", message=u"�G���[1�Fini�t�@�C���Ƀx���_�[��񂪋L�ڂ���Ă��܂���")
        return -1
    if int(parent.radio2.get()) == 0:
        if not cp.get("Vender", "Type") == "HUA":
            messagebox.showerror(title=u"ini�t�@�C���ǂݍ��݃G���[", message=u"�G���[�Q�F�x���_�[���g�t�`�ɐݒ肳��Ă��܂�")
            return -1
    elif int(parent.radio2.get()) == 1:
        if not cp.get("Vender", "Type") == "ZTE":
            messagebox.showerror(title=u"ini�t�@�C���ǂݍ��݃G���[", message=u"�G���[�R�F�x���_�[���y�s�d�ɐݒ肳��Ă��܂�")
            return -1
    else:
        messagebox.showerror(title=u"ini�t�@�C���ǂݍ��݃G���[", message=u"�G���[�S�Fini�t�@�C���̃x���_�[���͂g�t�`���y�s�d�ɐ�������Ă��܂�")
        return -1
    global config2t
    global TabSort
    config2t = show_config(cp)
    TabSort = []
    for a in config2t.keys():
        seq = str(config2t[a]['Tab01']['Tab01']).strip().split('-')
        if len(seq) == 2 or len(seq) == 3:
            if len(seq) == 3:
                seqkey = (('000' + seq[0]).strip()[-2:]) + '-' + (('000' + seq[1]).strip()[-2:]) + '-' + (
                    ('000' + seq[2]).strip()[-2:])
            else:
                seqkey = (('000' + seq[0]).strip()[-2:]) + '-' + (('000' + seq[1]).strip()[-2:]) + '-' + '01'
            seqkey = str(seqkey).strip()
            config2t[a]['Tab01']['Tab01'] = seqkey
            TabSort.append(list([config2t[a]['Tab01']['Tab01'], config2t[a]['Grp01']['Grp01'], a]))
    TabSort.sort()
    tabnames = []
    tabcontents = []
    wtag = ''
    for i in range(len(TabSort)):
        if wtag != TabSort[i][0][0:2]:
            templist = []
            k = i
            while k < len(TabSort) and TabSort[i][0][0:3] == TabSort[k][0][0:3]:
                templist.append(('cl' + TabSort[k][0][0:8], TabSort[k][0][0:8] + '_' + TabSort[k][1]))
                l = k + 0
                while l < len(TabSort) and TabSort[k][0][0:8] == TabSort[l][0][0:8]:
                    templist.append((['cl' + TabSort[l][0][0:8], TabSort[l][2]], TabSort[l][2]))
                    l = l + 1
                k = l
            tabnames.append("D" + TabSort[i][0][0:2].lstrip('0'))
            tabcontents.append(templist)
            wtag = TabSort[i][0][0:2]
    parent.inifileshort = Filemngr.getfile(filename)
    parent.l1.configure(text="   " + parent.inifileshort)
    parent.inifilename = filename
    return tabnames, tabcontents


def menu_load(menu, filename):
    """
    �T�u��ʁFini�t�@�C���I�����j���[�o�[�̐ݒ�
    :param menu:object
    :param filename:string
    :return:
    """
    master = menu.master  # current window
    parent = menu.arg  # parent window
    tabnames, tabcontents = get_tabs(menu, filename)
    tview = TabView(master)
    trees = [TreeView(tview.newtab(tabnames[i]), tabcontents[i], f_init=tree_init, f_select=tree_select) for i in
             range(len(tabnames))]
    if not menu.entryexists("Select"):
        selectmenu = menu.newmenu("Select")
        selectmenu.add_command(label="Select all in current tab",
                               command=lambda: trees[tview.getcurrenttabindex()].selectAll("on"))
        selectmenu.add_command(label="Select all", command=lambda: [t.selectAll("on") for t in trees])
        selectmenu.add_command(label="Deselect all in current tab",
                               command=lambda: trees[tview.getcurrenttabindex()].selectAll("off"))
        selectmenu.add_command(label="Deselect all", command=lambda: [t.selectAll("off") for t in trees])
    else:
        selectmenu = menu.getmenu("Select")
        selectmenu.entryconfig("Select all in current tab",
                               command=lambda: trees[tview.getcurrenttabindex()].selectAll("on"))
        selectmenu.entryconfig("Select all", command=lambda: [t.selectAll("on") for t in trees])
        selectmenu.entryconfig("Deselect all in current tab",
                               command=lambda: trees[tview.getcurrenttabindex()].selectAll("off"))
        selectmenu.entryconfig("Deselect all", command=lambda: [t.selectAll("off") for t in trees])


def menu_save(menu, filename):
    """
    �T�u��ʁFini�t�@�C���I��save�R�}���h�̎��s
    :param menu:object
    :param filename:string
    :return:
    """
    master = menu.master  # current window
    parent = menu.arg  # parent window
    global config2t
    global TabSort
    f = open(filename, 'w')
    f.write("[Vender]\n")
    if int(parent.radio2.get()) == 0:
        f.write("Type: HUA\n")
    elif int(parent.radio2.get()) == 1:
        f.write("Type: ZTE\n")
    for i in range(len(TabSort)):
        # print(config2t[TabSort[i][2]])
        f.write("[" + TabSort[i][2] + "]" + "\n")
        s = ""
        s = s + "Report:" + config2t[TabSort[i][2]]["Report"]["Report"] + "\n"
        s = s + "CriCfg:" + config2t[TabSort[i][2]]["CriCfg"]["CriCfg"] + "\n"
        s = s + "CriVal:" + config2t[TabSort[i][2]]["CriVal"]["CriVal"] + "\n"
        s = s + "AxisTitle:" + config2t[TabSort[i][2]]["AxisTitle"]["AxisTitle"] + "\n"
        s = s + "MinimumScale:" + config2t[TabSort[i][2]]["MinimumScale"]["MinimumScale"] + "\n"
        s = s + "MinimumScale Auto:" + config2t[TabSort[i][2]]["MinimumScale Auto"]["MinimumScale Auto"] + "\n"
        s = s + "MaximumScale:" + config2t[TabSort[i][2]]["MaximumScale"]["MaximumScale"] + "\n"
        s = s + "MaximumScale Auto:" + config2t[TabSort[i][2]]["MaximumScale Auto"]["MaximumScale Auto"] + "\n"
        s = s + "MajorUnit:" + config2t[TabSort[i][2]]["MajorUnit"]["MajorUnit"] + "\n"
        s = s + "MajorUnit Auto:" + config2t[TabSort[i][2]]["MajorUnit Auto"]["MajorUnit Auto"] + "\n"
        s = s + "Grp01:" + config2t[TabSort[i][2]]["Grp01"]["Grp01"] + "\n"
        s = s + "Tab01:" + config2t[TabSort[i][2]]["Tab01"]["Tab01"] + "\n"
        s = s + "KpiFlg01:" + config2t[TabSort[i][2]]["KpiFlg01"]["KpiFlg01"] + "\n"
        s = s.replace("%", "%%")
        f.write(s)
    f.close
    parent.inifileshort = Filemngr.getfile(filename)
    parent.l1.configure(text="   " + parent.inifileshort)
    parent.inifilename = filename


def tree_init(tree, item, itemarr):
    """
    �T�u��ʁFini�t�@�C���I�������c���[�\��
    :param tree:object
    :param item:string
    :param itemarr:string
    :return:
    """
    global config2t
    if len(itemarr) == 2:
        if config2t[itemarr[-1]]['KpiFlg01']["KpiFlg01"] == "0":
            tree.setstatus(item, "off")
        else:
            tree.setstatus(item, "on")
            parent = tree.hlist.info_parent(item)
            tree.setstatus(parent, "on")


def tree_select(tree, item, itemarr):
    """
    �T�u��ʁFini�t�@�C���I�������c���[�I��

    :param tree:object
    :param item:
    :param itemarr:string
    :return:
    """
    global config2t
    if len(itemarr) == 2:
        if tree.getstatus(item) == 'on':
            # print("on:  " + item)
            config2t[itemarr[-1]]['KpiFlg01']["KpiFlg01"] = "1"
        else:
            # print("off: " + item)
            config2t[itemarr[-1]]['KpiFlg01']["KpiFlg01"] = "0"


class Application(tk.Frame):
    """
    ���C����ʃR���g���[���z�u

    txBox1�{openb1      :text+button     :���͂j�o�h�t�H���_�[�{�I��
    txBox2�{openb2      :text+button     :�o�̓O���t�t�H���_�[�{�I��
    rbtn2               :radio           :HUA or ZTE �I��
    nwb                 :button          :�O���t�ڍבI��
    rbtn4               :radio           :�������[�h�I��  u'���f�����O�{�O���t', u'���f�����O', u'�O���t'
    rbtn                :radio           :��͊��ԑI��    u'3 Days', u'1 Week', u'Free'
    s_pane              :text            :�J�n��
    e_pane              :text            :�I����
    stbtn               :button          :�X�^�[�g�{�^��
    pb1                 :progress        :�v���O���X�o�[
    """
    def __init__(self, master=None):
        """

        :param master:object
        """
        super().__init__(master)
        w = 500
        h = 600
        x = (self.master.winfo_screenwidth() - w) / 2
        y = (self.master.winfo_screenheight() - h) / 2
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        root.configure(background="white")

        style = ttk.Style()
        style.configure("TRadiobutton", foreground=C_BLUE, background="white")
        style.configure("TEntry", foreground=C_BLUE, highlightbackground=C_BLUE)
        style.configure("TPanedwindow", foreground=C_BLUE, background="white")
        style.configure("TLabel", foreground=C_BLUE, background="white")

        style.configure("inv.TPanedwindow", foreground="white", background=C_BLUE)
        style.configure("inv.TLabel", foreground="white", background=C_BLUE)

        root.update()
        root.resizable(width=False, height=False)

        # ���͂j�o�h�p�^�[��
        # �~�σf�[�^�t�H���_�[
        self.strpath2 = parentdir + "/ref_KPI_Data"
        self.temppath = maindir + "/temp"
        self.backuppath = maindir + "/BackUp"
        self.learningpath = maindir.replace(chr(165), "/") + "/Learning_Data"
        # self.learningpath = 'C:/Users/wcp/Anaconda3/Scripts' + "/Learning_Data"
        self.default_inipath = parentdir + "\\ini"
        self.default_inipath_hua = self.default_inipath + "\\HUA_�S�I��.ini"
        self.default_inipath_zte = self.default_inipath + "\\ZTE_�S�I��.ini"

        self.strrpath = '.*(ZTE|HUA).*_.*.csv'
        # �~�σf�[�^�t�H���_�[
        if os.path.exists(self.strpath2) == False:
            os.mkdir(self.strpath2)
        if os.path.exists(self.temppath) == False:
            os.mkdir(self.temppath)
        if os.path.exists(self.backuppath) == False:
            os.mkdir(self.backuppath)
        if os.path.exists(self.learningpath) == False:
            os.mkdir(self.learningpath)
        self.pack()
        self.radio = tk.StringVar()
        self.radio2 = tk.StringVar()
        self.radio4 = tk.StringVar()

        self.path = tk.StringVar()
        self.labeltxt = tk.StringVar()
        self.rframe = ttk.Frame(master)

        self.toppane = ttk.PanedWindow(root)
        s = ttk.Style()
        s.configure("toppane.TPanedwindow", foreground=C_BLUE, background=C_BLUE)
        self.toppane.configure(style="toppane.TPanedwindow")
        self.toppane.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # ���x��
        self.lblpane1 = LabelPane(self.toppane, u'KPI�f�[�^�I��', "white", C_BLUE)
        # �e�L�X�g�{�b�N�X
        self.txBox1 = ttk.Entry(self.lblpane1.botpane, width=20)
        self.txBox1.insert(tk.END, "")
        self.txBox1.pack(side=tk.LEFT, padx=5, expand=1, fill=tk.X)
        # �_�C�A���O�{�b�N�X
        self.openb = tk.Button(self.lblpane1.botpane, bg="white", fg=C_BLUE, relief='groove', text='Open',
                               command=self.opendir, width=10)
        self.openb.pack(side=tk.RIGHT, padx=5)
        self.lblpane1.pack(fill=tk.X, expand=1)
        self.txBox1.bind("<KeyRelease>", self.txbox1_text)

        # ���x��
        self.lblpane2 = LabelPane(self.toppane, u'�O���t�o�͐�I��', "white", C_BLUE)
        # �e�L�X�g�{�b�N�X
        self.txBox2 = ttk.Entry(self.lblpane2.botpane, width=20)
        self.txBox2.insert(tk.END, "")
        self.txBox2.pack(side=tk.LEFT, padx=5, expand=1, fill=tk.X)
        # �_�C�A���O�{�b�N�X
        self.openb2 = tk.Button(self.lblpane2.botpane, bg="white", fg=C_BLUE, relief='groove', text='Open',
                                command=self.opendir2, width=10)
        self.openb2.pack(side=tk.RIGHT, padx=5)
        self.lblpane2.pack(fill=tk.X, expand=1)
        self.txBox2.bind("<KeyRelease>", self.txbox2_text)

        # �x���_�[�I��
        self.i1 = ImagePane(root, i1r)
        self.i1.pack(anchor=tk.W)
        self.midlblpane0 = LabelPane(self.i1.rpane, u'�x���_�[', C_BLUE, "white")
        self.midlblpane0.pack()
        self.rbtn2 = []
        self.radio2.set(0)
        for i, x in enumerate((u'HUA', u'ZTE')):
            self.rbtn2.append(ttk.Radiobutton(self.midlblpane0.botpane, text=x, value=i, variable=self.radio2,
                                              state=tk.NORMAL, command=self.set_ini_default))
            self.rbtn2[i].pack(side=tk.LEFT)
        # �Ώ�KPI�I��
        self.i2 = ImagePane(root, i2r)
        self.i2.pack(anchor=tk.W)
        self.midlblpane1 = LabelPane(self.i2.rpane, u'�Ώ�KPI�ݒ�', C_BLUE, "white")
        self.midlblpane1.pack()
        self.nwb = NewWindowButton(self.midlblpane1.botpane, "�I��", title="�O���t�ڍבI��", w=500, h=500, fg=C_BLUE, bg="white")
        self.inifilename = ""
        self.inifileshort = ""
        self.nwb.f_on_create = lambda: createnewwindow(self.nwb, self)
        self.nwb.b_create = lambda: [b.configure(state=tk.DISABLED) for b in self.rbtn2]
        self.nwb.b_close = lambda: [b.configure(state=tk.NORMAL) for b in self.rbtn2]
        self.nwb.pack(side=tk.LEFT)
        self.l1 = ttk.Label(self.midlblpane1.botpane, text=" ")
        self.l1.pack(side=tk.LEFT)

        # �������[�h�I��
        self.i3 = ImagePane(root, i3r)
        self.i3.pack(anchor=tk.W)
        self.midlblpane2 = LabelPane(self.i3.rpane, u'�������[�h', C_BLUE, "white")
        self.midlblpane2.pack()
        self.rbtn4 = []
        self.radio4.set(0)
        for i, x in enumerate((u'���f�����O�{�O���t', u'���f�����O', u'�O���t')):
            self.rbtn4.append(ttk.Radiobutton(self.midlblpane2.botpane, text=x, value=i, variable=self.radio4,
                                              state=tk.NORMAL))
            self.rbtn4[i].pack(side=tk.LEFT)

        # ��͊��ԑI��
        self.i4 = ImagePane(root, i4r)
        self.i4.pack(anchor=tk.W)
        self.midlblpane3 = LabelPane(self.i4.rpane, u'��͊���', C_BLUE, "white")
        self.midlblpane3.pack(side=tk.LEFT)
        self.rbtn = []
        self.radio.set(0)
        for i, x in enumerate((u'3 Days', u'1 Week', u'Free')):
            self.rbtn.append(
                ttk.Radiobutton(self.midlblpane3.botpane, text=x, value=i, variable=self.radio, state=tk.NORMAL,
                                command=self.radio_state))
            self.rbtn[i].pack(side=tk.LEFT)

        # [�J�n��,�I����]���O���[�v����E�B���h�E�y�C��
        self.datepane = ttk.PanedWindow(self.midlblpane3.botpane, orient=tk.VERTICAL, style="inv.TPanedwindow")
        self.datepane.pack(anchor=tk.W, expand=1)

        # �e�L�X�g�{�b�N�X
        self.s_pane = DatePane(self.datepane, u'�@�J�n��', C_BLUE, "white", fg_enable=C_BLUE)
        self.e_pane = DatePane(self.datepane, u'�@�I����', C_BLUE, "white", fg_enable=C_BLUE)
        self.radio_state()

        # �X�^�[�g�{�^��
        self.startpane = ttk.PanedWindow(root)
        self.startpane.pack(expand=1)

        # ���y�C��
        self.startpane_l = ttk.PanedWindow(self.startpane)
        self.stbtn = tk.Button(self.startpane_l, text="START", state=tk.DISABLED, command=self.start_btn, bg="white",
                               fg=C_BLUE, relief='groove', height=2, width=12, highlightthickness=1,
                               highlightbackground=C_BLUE, highlightcolor=C_BLUE)
        self.stbtn.pack()
        self.startpane_l.pack(side=tk.LEFT, padx=5)

        # ���y�C��
        self.startpane_r = ttk.PanedWindow(self.startpane)
        self.startpane_r.pack(side=tk.RIGHT, padx=5)
        self.pb1 = Meter(self.startpane_r)

        self.startpane_d = DebugPane(self.startpane_r, lambda self=self: self.f_debug())
        self.startpane_d.pack(side=tk.BOTTOM)

        self.rframe.pack(pady=10)
        self.set_ini_default()

    def f_debug(self):
        """

        :return:
        """
        sd["debug"] = True
        print("sd debug mode on")
        self.slider_tpara = Slider(self.startpane_d, power=-2, from_=1.0, to=20.0, value=1.05, text="tpara")
        self.slider_epochs = Slider(self.startpane_d, power=0, from_=50, to=1000, value=100, text="epoch")
        self.slider_procs = Slider(self.startpane_d, power=0, from_=1, to=100, value=3, text="procs")

    # �x���_�[�I���ɂ��f�t�H���g�Ώۂj�o�h�t�@�C���ݒ�
    def set_ini_default(self):
        """
        �x���_�[�ʃf�t�H���gINI�p�X�ݒ�
        :return:
        """
        if int(self.radio2.get()) == 0:
            self.set_inifilepath(self.default_inipath_hua)
        else:
            self.set_inifilepath(self.default_inipath_zte)

    # �Ώۂj�o�h�t�@�C���ݒ�
    def set_inifilepath(self, filename):
        """
        �f�t�H���gINI�t�@�C���Ǎ���
        :param filename:string
        :return:
        """
        if os.path.exists(filename):
            self.inifileshort = Filemngr.getfile(filename)
            self.l1.configure(text="   " + self.inifileshort)
            self.inifilename = filename
            global config2t
            ini2 = ConfigParser.SafeConfigParser(strict=False)
            ini2.optionxform = str
            ini2.read(filename)
            config2t = show_config(ini2)
        else:
            self.inifileshort = ""
            self.l1.configure(text=" ")
            self.inifilename = ""
            messagebox.showerror(title=u"ini�t�@�C���ǂݍ��݃G���[", message=u"�G���[�T�Fini�t�@�C��" + filename + u"��������܂���")

    # ���b�Z�[�W�̕\��
    def message_dest(self):
        """
        �f�t�H���gINI�t�@�C���Ǎ���

        :return:
        """
        self.subWindow.destroy()

    def message_open(self):
        """
        ���b�Z�[�W�̕\��
        :return:
        """
        self.subWindow = tk.Toplevel()
        self.subWindow.title("���b�Z�[�W")
        sublabel = tk.Label(self.subWindow, text=self.msg)
        sublabel.pack()

    def txbox1_text(self, event):
        """
        KPI���͓��̓t�H���_�[�I���̃C�x���g

        :param event:event
        :return:
        """
        if self.txBox1.get() != "" and self.txBox2.get() != "":
            self.stbtn.configure(foreground="white", background=C_GOLD)
            self.stbtn_state(tk.NORMAL)
        else:
            self.stbtn.configure(foreground=C_BLUE, background="white")
            self.stbtn_state(tk.DISABLED)

    # �e�L�X�g�{�b�N�X�̃C�x���g
    def txbox2_text(self, event):
        """
        �O���t�o�̓t�H���_�[�I���̃C�x���g

        :param event:event
        :return:
        """
        if self.txBox1.get() != "" and self.txBox2.get() != "":
            self.stbtn.configure(foreground="white", background=C_GOLD)
            self.stbtn_state(tk.NORMAL)
        else:
            self.stbtn.configure(foreground=C_BLUE, background="white")
            self.stbtn_state(tk.DISABLED)

    def opendir(self):
        """
        KPI���͓��̓t�H���_�[�I���̃{�^��

        :return:
        """
        dirname = tkfd.askdirectory(title=u'Kpi�f�[�^�I��')
        if dirname != "":
            self.txBox1.delete(0, tk.END)
            self.txBox1.insert(0, dirname)
        if self.txBox1.get() != "" and self.txBox2.get() != "":
            self.stbtn.configure(foreground="white", background=C_GOLD)
            self.stbtn_state(tk.NORMAL)
        else:
            self.stbtn.configure(foreground=C_BLUE, background="white")
            self.stbtn_state(tk.DISABLED)

    def opendir2(self):
        """
        �O���t�o�̓t�H���_�[�I���̃{�^��

        :return:
        """
        dirname = tkfd.askdirectory(title=u'�o�̓f�[�^�I��')
        if dirname != "":
            self.strpath4 = os.path.dirname(dirname)
            self.txBox2.delete(0, tk.END)
            self.txBox2.insert(0, dirname)
        if self.txBox1.get() != "" and self.txBox2.get() != "":
            self.stbtn.configure(foreground="white", background=C_GOLD)
            self.stbtn_state(tk.NORMAL)
        else:
            self.stbtn.configure(foreground=C_BLUE, background="white")
            self.stbtn_state(tk.DISABLED)

    def txBox1_state(self, stat):
        """
        KPI���̓t�H���_�[�I���̃e�L�X�g�g�p�ې؂�ւ�

        :param stat:int
        :return:
        """
        self.txBox1["state"] = stat

    def txBox2_state(self, stat):
        """
        �O���t�o�̓t�H���_�[�I���̃e�L�X�g�g�p�ې؂�ւ�

        :param stat:int
        :return:
        """
        self.txBox2["state"] = stat

    def radio_state(self):
        """
        �O���t�o�͊��ԑI�����W�I�{�^������
        :return:
        """
        if int(self.radio.get()) == 2:
            self.s_pane.enable()
            self.e_pane.enable()
        else:
            self.s_pane.disable()
            self.e_pane.disable()

    def radio_state2(self, stat):
        """
        �O���t�o�͊��ԑI�����W�I�{�^������
        :param stat:int
        :return:
        """
        for rb in self.rbtn:
            rb["state"] = stat
        if int(self.radio.get()) == 2 and stat == tk.NORMAL:
            self.s_pane.enable()
            self.e_pane.enable()
        else:
            self.s_pane.disable()
            self.e_pane.disable()

    def radio2_state2(self, stat):
        """
        �x���_�[�I���g�p�ېݒ�
        :param stat:int
        :return:
        """
        for rb in self.rbtn2:
            rb["state"] = stat

    def radio4_state2(self, stat):
        """
        �o�̓��[�h�I���g�p��
        :param stat:int
        :return:
        """
        for rb in self.rbtn4:
            rb["state"] = stat

    def stbtn_state(self, stat):
        """
        �X�^�[�g�{�^���g�p��
        :param stat:
        :return:
        """
        self.stbtn["state"] = stat

    def stbtn_disable(self):
        """
        �X�^�[�g:�{�^���g�p�s��
        :return:
        """
        self.nwb.disable()
        if sd["debug"]:
            self.slider_tpara.disable()
            self.slider_epochs.disable()
            self.slider_procs.disable()
        for f in [self.txBox1_state, self.txBox2_state, self.stbtn_state, self.radio_state2, self.radio2_state2,
                  self.radio4_state2]:
            f(tk.DISABLED)
        self.stbtn.configure(foreground=C_BLUE, background="white")

    def stbtn_reset(self):
        """
        �X�^�[�g�{�^��������
        :return:
        """
        # clean up
        for t in tempfiles:
            os.remove(t)
        tempfiles.clear()
        self.nwb.enable()
        if sd["debug"]:
            self.slider_tpara.enable()
            self.slider_epochs.enable()
            self.slider_procs.enable()
        for f in [self.txBox1_state, self.txBox2_state, self.stbtn_state, self.radio_state2, self.radio2_state2,
                  self.radio4_state2]:
            f(tk.NORMAL)
        self.stbtn.configure(text="START", foreground="white", background=C_GOLD, command=self.start_btn)
        self.pb1.update_bar(0)
        self.pb1.update_lbl("Progress message")

    def stbtn_fin(self):
        """
        �X�^�[�g�����I����
        :return:
        """
        self.stbtn_reset()
        if len(self.ijyoufiles) == 0:
            d = Dialogbox(self.master, "Finished", u"\n�������������܂���\n\nKPI�ُ�Ȃ�", image=i2r)
            d.centerwindow()
        else:
            d = Dialogbox(self.master, "Finished", u"\n�������������܂���\n\nKPI�ُ킠��\n�o�̓t�H���_���m�F���Ă�������", image=i2r)
            for file in self.ijyoufiles:
                folder = Filemngr.getfolder(file)
                filename = Filemngr.getfile(file)
                d.inserttext(filename,
                             lambda folder=folder, file=file: Filemngr.opendir(folder) or Filemngr.opennote(file))
                d.inserttext("\n")
            d.insertend()
            d.centerwindow()

    def updatepb(self):
        """
        �i�s�󋵂̍X�V
        :return:
        """
        self.counter_done = sd["counter"]
        self.pb1.update_bar(int(100 * ((gui.donecount + self.counter_done) / (gui.taskcount + self.counter_task))))
        self.pb1.update_lbl(
            str(gui.donecount + self.counter_done) + "/" + str(gui.taskcount + self.counter_task) + " tasks completed")

    # cancel button pressed
    def on_cancel(self):
        """
        �L�����Z���{�^��������
        :return:
        """
        self.pb1.update_lbl("Canceling...")
        self.stbtn.configure(state=tk.DISABLED, foreground=C_BLUE, background="white")
        sd["exit"] = True  # �T�u�v���Z�X�I���t���O
        gui.thread_stop()  # �X���b�h�I���t���O

    def show_growth(self):
        """

        :return:
        """
        import objgraph
        objgraph.show_growth()

    def start_btn(self):
        """
        �X�^�[�g�{�^��������
        :return:
        """
        try:
            if self.nwb.isopen():
                messagebox.showerror(title=u"KPI�ُ픻��c�[�� �G���[", message=u"�G���[�U�F�O���t�ڍבI������Ă���X�^�[�g�������ĉ�����")
                return
            gui.reset()
            self.ijyoufiles = set()
            self.stbtn_disable()
            global tpara
            global epochs
            global procs
            if sd["debug"]:
                tpara = self.slider_tpara.value
                epochs = self.slider_epochs.value
                procs = self.slider_procs.value
            print("running with tpara:", tpara)
            print("running with epochs:", epochs)
            print("running with procs:", procs)
            self.pb1.update_lbl("Checking data...")
            msg = self.datachk()
            if msg != None:
                messagebox.showerror(title=u"KPI�ُ픻��c�[�� �G���[", message=msg)
                self.stbtn_reset()
                return
            # �����J�n
            os.chdir(self.strpath2)
            self.counter_done = 0
            self.counter_task = 0
            sd["exit"] = False
            sd["counter"] = 0
            os.chdir(self.learningpath)
            for i, fullfile in enumerate(self.files):  # �j�o�h�f�[�^�P��Loop
                gui.thread_start(lambda self=self, fullfile=fullfile, i=i: self.setpaths(fullfile, i),
                                 lambda self=self: self.updatepb())
                gui.thread_start(lambda self=self: self.fmtconv(),
                                 lambda self=self: self.updatepb())  # KPI+�~�σf�[�^�̓ǂݍ��݃��x���t��
                gui.thread_start(lambda self=self: self.pool(), lambda self=self: self.updatepb())
                gui.thread_start(lambda self=self: self.outputdf(), lambda self=self: self.updatepb())
                # gui.thread_start(lambda self=self:self.show_growth(),lambda self=self:self.updatepb())                           #memory debug
            gui.thread_start(lambda self=self: self.stbtn_fin())
            self.stbtn.configure(foreground="white", background=C_GOLD, text="CANCEL",
                                 command=lambda self=self: self.on_cancel())
            self.stbtn_state(tk.NORMAL)
        except Exception as e:
            gui.reset()
            self.stbtn_reset()
            ThreadingException.msgbox()

    def datachk(self):
        """
        ���̓`�F�b�N����
        :return:
        """
        # �~�σf�[�^�z���_�[�ɍ�Ɨp�z���_�[�쐬
        if os.path.exists(self.temppath) == False:
            os.mkdir(self.temppath)
        if os.path.exists(self.backuppath) == False:
            os.mkdir(self.backuppath)

        # ��ʓ��̓`�F�b�N
        self.pb1.update_lbl("checking data...")
        self.strpath = self.txBox1.get()
        self.strpath4 = self.txBox2.get()
        if os.path.isdir(self.strpath) == False:
            return u"�G���[�V�F��͑Ώۃf�[�^�t�H���_" + self.strpath + u"������܂���"
        if os.path.isdir(self.strpath4) == False:
            return u"�G���[�W�F�O���t�o�͐�t�H���_" + self.strpath4 + u"������܂���"
        if os.path.isdir(self.strpath2) == False:
            return u"�G���[�X�F�j�o�h�~�ϗp�t�H���_" + self.strpath2 + u"������܂���"
        os.chdir(self.strpath)

        # ���t���̓`�F�b�N
        if int(self.radio.get()) == 2:


            self.sdatetime = self.s_pane.getdatetime()
            self.edatetime = self.e_pane.getdatetime()

            if self.sdatetime is None:
                return u"�G���[�P�W�F�J�n���t�G���["
            if self.edatetime is None:
                return u"�G���[�P�X�F�I�����t�G���["
            self.sdatetime = self.sdatetime.replace(hour=0)
            self.edatetime = self.edatetime.replace(hour=23)
            if self.sdatetime > self.edatetime:
                return u"�G���[�Q�O�F�J�n���t>�I�����t�G���["

        filesg = os.listdir(self.strpath)
        self.files = []
        # �Ώۂj�o�h�t�@�C���Z���N�g
        if int(self.radio2.get()) == 0:
            self.strrpath = '.*(HUA).*_.*.csv'
        else:
            self.strrpath = '.*(ZTE).*_.*.csv'

        count = 0
        for file in filesg:
            mt = re.match(self.strrpath, file)
            if hasattr(mt, "group"):
                self.files.append(mt.group(0))
                count = count + 1
        if count == 0:
            return u"�G���[�P�O�F��͑Ώۃf�[�^�t�H���_" + self.strpath + u"���ɊY������t�@�C��������܂���"

        self.tmpfile = []
        self.ofile = []

        # �Ώۂj�o�h�ʃ`�F�b�N
        for k, file in enumerate(self.files):  # ��͑Ώۃf�[�^�����[�v
            # KPI�f�[�^�ǂݍ���
            df = CSV.read(self.strpath + "/" + file, header=0, tempdir=self.temppath, tempfile="tmp_sjis.csv", nrows=1)
            is_empty = True
            for kpi in config2t.keys():
                if int(config2t[kpi]['KpiFlg01']['KpiFlg01']) == 1 and kpi in df.columns:
                    is_empty = False
            if is_empty:
                return u"�G���[�P�P�F��͑Ώۃf�[�^" + self.strpath + "/" + file + u"���ɊY������j�o�h������܂���"

            f1o = open(self.strpath + "/" + file, 'r')
            f1 = csv.reader(f1o, delimiter=',')
            test_data = []
            i = 0
            j = 0
            for row in f1:
                if i == 0 or j > 0:
                    test_data.append(row)
                if i != 0 and j > 0:
                    test_data[i][0] = test_data[i][0].replace("/", "-")
                if i == 0 or j > 168:
                    i = i + 1
                j = j + 1
            f1o.close
            if len(test_data) < 2:
                return u"�G���[�P�Q�F��͑Ώۃf�[�^" + self.strpath + "/" + file + u"�Ƀf�[�^������܂���"
            for i in range(2, len(test_data)):
                t1 = todatetime(test_data[i - 1][0])
                t2 = todatetime(test_data[i][0])
                # �d���f�[�^�m�F
                wksa = (t2 - t1).seconds
                if wksa == 0:
                    return u"�G���[�P�R�F��͑Ώۃf�[�^" + self.strpath + "/" + file + u"��" + test_data[i - 1][0] + u"���d�����Ă��܂�"
            # �~�σf�[�^����
            vend = re.search('(ZTE|HUA)', file)
            if vend:
                vend.start
                # (ZTE|HUA)_xxxxx�Z�o
                ofile = file[vend.start():]
                self.tmpfile.append(file[vend.start():])
                filesg = os.listdir(self.strpath2)
                fcnt = 0
                for file2 in filesg:
                    mt = re.match('.*' + self.tmpfile[k].replace("(", "\(").replace(")", "\)"), file2)
                    if hasattr(mt, "group"):
                        self.ofile.append(file2)
                        fcnt = fcnt + 1
                if fcnt == 0:
                    return u"�G���[�P�T�F�~�σf�[�^�t�H���_" + self.strpath2 + u"���ɉ�͑Ώۃf�[�^�ƈ�v����t�@�C��������܂���"
                if fcnt > 1:
                    return u"�G���[�P�U�F�~�σf�[�^�t�H���_" + self.strpath2 + u"�ɏd���t�@�C��������܂�"
            else:
                return u"�G���[�P�S�F��͑Ώۃf�[�^" + file + u"�̃t�@�C������HUA�EZTE���t���Ă��܂���"

            # �~�σf�[�^�ǂݍ���
            f2o = open(self.strpath2 + "/" + self.ofile[k], 'r')
            f2 = csv.reader(f2o, delimiter=',')
            train_data = []
            i = 0
            for row in f2:
                train_data.append(row)
                if i != 0:
                    train_data[i][0] = train_data[i][0].replace("/", "-")
                i = i + 1
            f2o.close

            if len(train_data) > 1:
                for i in range(2, len(train_data)):
                    t1 = todatetime(train_data[i - 1][0])
                    t2 = todatetime(train_data[i][0])
                    # �d���f�[�^�m�F
                    wksa = (t2 - t1).seconds
                    if wksa == 0:
                        msg = u"�G���[�P�V�F�~�σf�[�^�t�@�C��" + self.strpath2 + "/" + self.ofile[k] + u"��" + train_data[i - 1][
                            0] + u"���d�����Ă��܂�"
                        return 1, msg

            # ��Ɨp�E�~�σt�@�C���̏o�́i�d���f�[�^�폜�j
            t1 = todatetime(test_data[1][0])
            if len(train_data) > 1:
                t2 = todatetime(train_data[-1][0])
            else:
                t2 = todatetime("2000-01-01 00:00:00")
            wksa = (t1 - t2).seconds
            tempfiles.add(self.temppath + "/tmp" + self.tmpfile[k])
            f = open(self.temppath + "/tmp" + self.tmpfile[k], 'w')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(train_data[0])  # �w�b�_�[����������

            # ��ʑI�����Ԃ̑Ó����`�F�b�N
            i = 1
            l = len(train_data)
            c = 0
            while l > i:
                if t1 > todatetime(train_data[i][0]):
                    writer.writerows(train_data[i:i + 1])
                    c = c + 1
                i = i + 1

            if c == 0:
                f.close()
                for t in tempfiles:
                    os.remove(t)
                tempfiles.clear()
                return u"�G���[�Q�Q�F�~�σf�[�^�t�@�C���Ƀf�[�^������܂���"

            if int(self.radio.get()) == 2:
                if self.edatetime < todatetime(test_data[1][0]):
                    return u"�G���[�Q�P�F�ŏI���t<�J�n�f�[�^���t�G���["
                if self.sdatetime > todatetime(test_data[-1][0]):
                    return u"�G���[�Q�P�F�J�n���t>�ŏI�f�[�^���t�G���["
        return None

    def setpaths(self, fullfile, i):
        """
        �֘A�t�H���_�[�̃`�F�b�N����э쐬
        :param fullfile:string
        :param i:int
        :return:
        """
        self.file = str(os.path.basename(fullfile))
        self.tmp = 'tmp' + self.tmpfile[i]
        self.bak = self.ofile[i]
        # �O���t�ۑ��f�B���N�g���[�쐬
        if int(self.radio2.get()) == 0:
            vend = re.search('(HUA)', self.file)
        else:
            vend = re.search('(ZTE)', self.file)
        # datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.strgpath = self.file[vend.start():-4]
        if os.path.exists(self.strpath4 + "/" + self.strgpath) == False:
            os.mkdir(self.strpath4 + "/" + self.strgpath)
        # ���f���ۑ��f�B���N�g���[�쐬
        if os.path.exists(self.learningpath + "/" + self.strgpath) == False:
            os.mkdir(self.learningpath + "/" + self.strgpath)
        dtnow = (str(pd.datetime.now())[0:-7]).replace(":", "-")  # yyyy-mm-dd hh:mm:ss.eeeeee to yyyy-mm-dd hh-mm-ss
        self.outpath = self.strpath4 + "/" + self.strgpath + "/" + dtnow
        os.mkdir(self.outpath)
        self.ijyoufile = self.outpath + "/" + self.strgpath + "_�ُ�.csv"
        f = open(self.ijyoufile, 'w')
        f.write("KPI��,�J�n����,�I������,���l\n")
        f.close()

    def fmtconv(self):
        """
        KPI �{ �g���[�j���O�f�[�^�̓Ǎ��݁@����уt�H�[�}�b�g����

        :return:
        """
        # �j�o�h�f�[�^�ǂݍ��݁|�|�|��numpy�ϊ�
        self.dfkpi = CSV.read(self.strpath + "/" + self.file, header=0, tempdir=self.temppath, tempfile="tmp_sjis.csv")
        if "Datetime" in self.dfkpi.columns:
            self.dfkpi.rename(columns={'Datetime': 'DateTime'}, inplace=True)
        self.dfkpi, _ = Time.df_fill(self.dfkpi)
        self.dfkpi['DateTime'] = pd.to_datetime(self.dfkpi['DateTime'])

        # �~�σf�[�^�ǂݍ��݁|�|�|��numpy�ϊ�
        self.dftrain = CSV.read(self.temppath + "/" + self.tmp, header=0, tempdir=self.temppath,
                                tempfile="tmp_sjis.csv")
        if "Datetime" in self.dftrain.columns:
            self.dftrain.rename(columns={'Datetime': 'DateTime'}, inplace=True)
        self.dftrain, idx = Time.df_fill(self.dftrain)
        self.dftrain['DateTime'] = pd.to_datetime(self.dftrain['DateTime'])
        lblcols = []
        for colname in self.dftrain.columns.tolist():
            if colname.endswith("_label"):
                lblcols.append(colname)
        self.dftrain.ix[idx, lblcols] = 9

        # ��KPI���J�b�g
        cut = 0
        for lb in self.dftrain.columns[0:min(10, len(self.dftrain))]:
            if lb in NOT_KPI:
                cut = cut + 1

        # ���x���ʒu����:�g���[�j���O���x����S�đޔ����폜
        wklbl = pd.DataFrame()
        for lb in self.dftrain.columns[cut:]:
            if lb[-6:] == "_label":
                wklbl[lb] = self.dftrain[lb]
                del self.dftrain[lb]
        for lb in self.dfkpi.columns[cut:]:
            if lb[-6:] == "_label":
                del self.dfkpi[lb]

        # ���x���ʒu����:�g���[�j���O���x�������߂Ė߂�
        for lb in self.dftrain.columns[cut:]:
            if lb[-6:] != "_label":
                self.dfkpi[lb + "_label"] = 9
                if len(idx) > 0 or not (lb + "_label" in wklbl.columns):
                    self.dftrain[lb + "_label"] = 9
                else:
                    self.dftrain[lb + "_label"] = wklbl[lb + "_label"]
        self.dfkpi = pd.concat([self.dftrain, self.dfkpi])
        self.dfkpi.reset_index(drop=True, inplace=True)

    def pool(self):
        """
        KPI�ʃ��C������
        :return:
        """
        kpis = []
        for kpi in config2t.keys():
            if int(config2t[kpi]['KpiFlg01']['KpiFlg01']) == 1 and kpi in self.dfkpi.columns:
                kpis.append(kpi)
        self.counter_task = len(kpis) * len(self.files) * 3
        self.updatepb()
        # �p�X�ݒ�
        outpath = self.outpath
        ijyoupath = self.outpath + "/" + self.strgpath + "_�ُ�"
        modelpath = self.learningpath + "/" + self.strgpath + "/"
        init_modelpath = self.learningpath + "/init.h5"
        # �t���O�ݒ�
        flag_model = ((int(self.radio4.get()) == 0 or int(self.radio4.get()) == 1))
        flag_graph = ((int(self.radio4.get()) == 0 or int(self.radio4.get()) == 2))
        if int(self.radio.get()) == 0:
            ixs = 24 * 3
            ixe = 0
            title = "(3 Days)"
        elif int(self.radio.get()) == 1:
            ixs = 24 * 8
            ixe = 0
            title = "(1 Week)"
        else:
            ixs = len(self.dfkpi[self.dfkpi['DateTime'] >= self.sdatetime])
            ixe = len(self.dfkpi[self.dfkpi['DateTime'] > self.edatetime])
            title = "(Free)" + (self.s_pane.getdatestr(sep="/") + "�`" + self.e_pane.getdatestr(sep="/"))
        # n = mp.cpu_count()-1
        n = procs
        if n <= 1:  # �v���Z�X�� == 1
            global defined_main_keras
            if not defined_main_keras:
                defined_main_keras = True
                keras_import()
            kpis_lbl = [v + "_label" for v in kpis]
            dftrain_partial = self.dftrain[["DateTime"] + kpis + kpis_lbl]
            dfkpi_partial = self.dfkpi[["DateTime"] + kpis + kpis_lbl]
            result = process_kpis(self, dftrain_partial, dfkpi_partial, init_modelpath, modelpath, flag_model,
                                  flag_graph, ixs, ixe, title, outpath, ijyoupath + ".csv", tpara, epochs, config2t, sd)
            if result is None:  # cancel pressed
                return
            df = result
            self.dfkpi = CSV.df_rplval(self.dfkpi, df)
            nrows = CSV.rows(ijyoupath + ".csv")
            if nrows > 1:
                self.ijyoufiles.add(ijyoupath + ".csv")
        else:
            # args�Efilepaths�쐬
            args = []
            filepaths = []
            kpi_split = Arr.split(kpis, n)
            for i, kpis in enumerate(kpi_split):
                kpis_lbl = [v + "_label" for v in kpis]
                dftrain_partial = self.dftrain[["DateTime"] + kpis + kpis_lbl]
                dfkpi_partial = self.dfkpi[["DateTime"] + kpis + kpis_lbl]
                args.append((None, dftrain_partial, dfkpi_partial, init_modelpath, modelpath, flag_model, flag_graph,
                             ixs, ixe, title, outpath, ijyoupath + str(i) + ".csv", tpara, epochs, config2t))
                filepaths.append(ijyoupath + str(i) + ".csv")
            results = Pooler.start_ps(process_kpis, self.updatepb, initializer, args, n=n, sd=sd)
            if results is None:  # cancel pressed
                return
            dfs = []
            for v in results:
                if v is not None:
                    dfs.append(v)
            if len(dfs) == 0:
                return
            df = pd.concat(dfs, axis=1, join='inner')
            self.dfkpi = CSV.df_rplval(self.dfkpi, df)
            nrows = CSV.concat_row(filepaths=[ijyoupath + ".csv"] + filepaths, outputpath=ijyoupath + ".csv",
                                   tempdir=self.temppath, tempfile="tmp_sjis.csv")
            if nrows > 1:
                self.ijyoufiles.add(ijyoupath + ".csv")
            for f in filepaths:
                if os.path.isfile(f):
                    os.remove(f)

    def outputdf(self):
        """
        new�~�σf�[�^�o��
        :return:
        """
        # �g���[�j���O�AKpi�f�[�^�A��
        # ���̃g���[�j���O�f�[�^���o�b�N�A�b�v
        shutil.move(self.strpath2 + "/" + self.bak, self.backuppath + "/" + self.bak)
        # �ŏI�̃g���[�j���O�������t�@�C�����ɕt�����ۑ�
        tstamp = self.dfkpi['DateTime'].iloc[-1].strftime("%Y-%m-%d-%H")
        CSV.write(self.dfkpi, self.strpath2 + "/" + tstamp + "_" + self.strgpath + ".csv")


if __name__ == '__main__':
    mp.freeze_support()
    sd = Pooler.shared_dict()
    argc = len(sys.argv)
    root = tix.Tk()
    root.title('KPI�ُ픻��c�[��')
    app = Application(master=root)
    # debug mode
    sd["debug"] = False
    if sd["debug"]:
        app.radio.set(1)  # 1 week
        app.radio2.set(0)  # HUA
        # app.radio2.set(1)  #ZTE
        app.set_inifilepath("C:/Users/wcp/Desktop/KPI�ُ팟�m�c�[��_ win64bit_V1.7/ini/HUA_�S�I��.ini")
        app.txBox1.insert(0, "C:/Users/wcp/Desktop/KPI�ُ팟�m�c�[��_ win64bit_V1.7/test/yoko_h1")
        app.txBox2.insert(0, "C:/Users/wcp/Desktop/KPI�ُ팟�m�c�[��_ win64bit_V1.7/test/yoko_h1")
        app.stbtn.configure(text="START", foreground="white", background=C_GOLD, command=app.start_btn, state="normal")
    gui = ThreadedGUI(root, f_stop=lambda app=app: app.stbtn_reset())
    app.mainloop()
