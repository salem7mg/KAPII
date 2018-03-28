"""
概要
    KPIデータの異常判定&グラフ出力処理を行う。


画面説明

    メイン
    txBox1＋openb1      :text+button     :入力ＫＰＩフォルダー＋選択
    txBox2＋openb2      :text+button     :出力グラフフォルダー＋選択
    rbtn2               :radio           :HUA or ZTE 選択
    nwb                 :radio           :グラフ詳細選択
    rbtn4　　　　　　　 :radio           :処理モード選択  u'モデリング＋グラフ', u'モデリング', u'グラフ'
    rbtn　　　　　　　　:radio           :解析期間選択    u'3 Days', u'1 Week', u'Free'
        s_pane          :text            :開始日
        e_pane          :text            :終了日
    stbtn               :button          :スタートボタン
    pb1                 :progress        :プログレスバー

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
    日付時刻タイプへ変換
    :param str:string：　yyyy/mm/dd hh:mm:ss or yyyy/mm/dd hh:mm
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

# 平滑化期間
day = 24
week = 7
hrs_in_week = 24 * 7

# GUI色
C_GOLD = "#EECD59"
C_BLUE = "#57A4C8"

# パス設定
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

tpara = 1.05  # 閾値の補正値
epochs = 100  # エポック数
procs = 3  # プロセス数
tempfiles = set()  # tempファイルのリスト
global gui  # threaded gui

# iniファイル関連
config2t = {}
TabSort = []

defined_main_keras = False


def keras_import():
    """
    keras_import および　autoencoderパラメータ設定
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
    KPI別MAX値ファイル出力
    :param path     :string
    :param str      :string
    :return:
    """
    with open(path, 'w') as f:
        f.write(str)


def read_TMAX(path):
    """
    KPI別MAX値読み込み
    :param path:string
    :return:
    """
    r = None
    with open(path, 'r') as f:
        r = float(f.read())
    return r


def isconst(x):
    """
    リストが全て同じ数字の場合true
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
    プロセスメイン処理
    :param gui_arg:
    :param dftrain:pd               トレーニング
    :param dfkpi:pd                 ＫＰＩデータ
    :param init_modelpath:string    モデル格納Path+ファイル名
    :param modelpath:string         モデル格納Path+ファイル名
    :param flag_model:bool          モデル作成有無
    :param flag_graph:bool          グラフ作成有無
    :param ixs:int                  開始年月日の位置
    :param ixe:int                  終了年月日の位置
    :param title:string             KPIタイトル
    :param outpath:string           グラフ格納Path
    :param ijyoufile:string         異常ファイル格納Path+ファイル名
    :param tpara:init               閾値の補正値
    :param epochs:init              epoch数
    :param config2t:dict            設定ファイル
    :param sd:dict                  内部ワーク
    :return:pd                      正常異常ラベル
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
                # データ整理
                start = time.time()
                idx_train, idx_even = get_idx_train(kpi, dftrain, dfkpi, idx_nan, imgpath, sd)
                TMAX = max(np.nanmax(dftrain[kpi]), np.nanmax(dfkpi[kpi]))  # 最大値算出
                TMIN = min(np.nanmin(dftrain[kpi]), np.nanmin(dfkpi[kpi]))
                if abs(TMIN) > abs(TMAX):
                    TMAX = TMIN
                flag_train = not (np.isnan(TMAX) or (TMAX == 0) or (idx_train is None))
                end = time.time()
                print(kpi, u"データ整理", end - start)

                if "lock" in globals():
                    lock.acquire()
                    sd["counter"] = sd["counter"] + 1
                    lock.release()
                else:
                    sd["counter"] = sd["counter"] + 1
                    gui_arg.updatepb()
                if sd["exit"]:
                    return None

                # モデル作成・ロード
                start = time.time()
                if flag_train:
                    flag_model_exists = (os.path.exists(modelfl + '.h5') and os.path.exists(modelfl + '.tmax'))
                    train_data = np.array([np.array(dfkpi[kpi][i:i + 24]) for i in idx_even])
                    # モデル作成
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
                        # モデルロード
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
                        # 閾値
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
                print(kpi, u"モデル作成・ロード", end - start)

                if "lock" in globals():
                    lock.acquire()
                    sd["counter"] = sd["counter"] + 1
                    lock.release()
                else:
                    sd["counter"] = sd["counter"] + 1
                    gui_arg.updatepb()
                if sd["exit"]:
                    return None
            # グラフ
            start = time.time()
            if flag_graph:
                dfkpi.ix[idx_nan, kpi] = np.nan
                graph5(kpi, dfkpi, ixs, ixe, title + kpi, flag_ijyou, flag_ae_load, flag_all_zero, imgpath, ijyoufile,
                       config2t, sd)
            end = time.time()
            print(kpi, u"閾値・グラフ", end - start)

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
	トレーニング正常位置取得
    :param kpi:string               ＫＰＩ名
    :param dftrain:pd               トレーニングデータ
    :param dfkpi:pd                 ＫＰＩデータ
    :param idx_nan:np               ＫＰＩデータのnanの位置
    :param imgpath:string　　　　　 グラフ出力Path+ファイル名
    :param sd:dict                  内部ワーク
    :return:idx_train, idx_even     24*7正常データスタートインデックス、複製インデックス
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
    #Arr.partition_spanはスタートインデックスを返す,idx_evenは複製のインデックス
    #全て（168）揃っていない場合nanを返す
    idx_train, idx_even, span = Arr.partition_span(p, 0, 24, hrs_in_week)
    if span != hrs_in_week:
        idx_train = None
        idx_even = None
    return idx_train, idx_even


def get_ans(kpi, dftrain, dfkpi, idx_train, autoencoder, modelfl, TMAX, tpara, config2t, imgpath, sd):
    """
    トレーニング　および　正常異常判定
    :param kpi:string                           ＫＰＩ名
    :param dftrain:pd                           トレーニングデータ
    :param dfkpi:pd                             ＫＰＩデータ
    :param idx_train:partition start indices    24*7正常データスタートインデックス
    :param autoencoder:autoencoder              autoencoderオブジェクト
    :param modelfl:string                       モデルファイル名
    :param TMAX:float                           ＫＰＩ最大値
    :param tpara:int                            閾値補正
    :param config2t:dict                        設定ファイル
    :param imgpath:string                       出力グラフPath + グラフ名
    :param sd:dict                              内部ワーク
    :return ans:numpy                           正常異常サイン
    """
    train_data0 = Arr.partition_split(dfkpi[kpi], 24, len(dftrain) - 1)
    test_data0 = np.array([np.array(dfkpi[kpi][i:i + 24]) for i in range(len(dftrain) - 24, len(dfkpi[kpi]) - 24)])
    x_train_out = autoencoder.predict(train_data0 / TMAX) * TMAX
    # 実値ー判定値の絶対値算出
    diff_train = []
    for i in range(len(x_train_out)):
        diff_train.append(abs(x_train_out[i][-1] - train_data0[i][-1]))
    diff_train = [diff_train[i] for i in idx_train]
    diff_train = np.array(diff_train)
    # 絶対値より移動平均＆閾値算出
    div0 = pd.ewma(diff_train, span=hrs_in_week)
    div1 = np.nanmax(div0[hrs_in_week:len(div0)])
    x_test_out = autoencoder.predict(test_data0 / TMAX) * TMAX
    # autoencod-real差分算出
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
    グラフ作成
    :param kpi:string                       ＫＰＩ名
    :param dfkpi:df                         ＫＰＩデータ
    :param ixs:int                          グラフ開始位置
    :param ixe:int                          グラフ終了位置
    :param title:string                     グラフタイトル
    :param flag_ijyou:bool                  異常有無
    :param flag_ae_load:bool                モデルロード有無
    :param flag_all_zero:bool               トレーニングデータALLzero
    :param imgpath:string                   グラフ出力Path+グラフ名
    :param ijyoufile:string                 グラフ出力Path+異常ファイル名
    :param config2t:dict                    設定ファイル
    :param sd:dict                          内部ワーク
    :return:
    """
    Mpltfont.set('IPAexGothic')
    n = len(dfkpi[kpi])
    x = dfkpi['DateTime'][n - ixs:n - ixe].tolist()
    y = dfkpi[kpi][n - ixs:n - ixe].tolist()
    z = dfkpi[kpi + "_label"][n - ixs:n - ixe].tolist()
    z = [1 if z[i] == 1 else 0 for i in range(len(z))]
    # 左軸判定（初期値、増分、最大値）
    # 最大、最小値算出
    lo = np.nanmin(y)
    hi = np.nanmax(y)
    ymid = (hi + lo) / 2
    ymin_1 = 0
    ymax_1 = 0
    ystep_12 = 0
    # 最大値を１．５倍にし最大値補正
    if hi >= lo and hi != 0:
        ymin_1 = lo
        ystep_1 = hi
        if hi > 0:
            ystep_1 = 10 ** (int(math.log10(hi * 1.5)))
        else:
            ystep_1 = 2
        ymax_1 = ((int)(hi * 1.5 / ystep_1) + 1) * ystep_1
    # 設定ファイルの値反映
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
    # 異常一覧ファイル書き込み
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
        f.write(kpi + ",,,未判定\n")
        f.close()
        return


def labeledt(ax, dt, dtlabel, y, c):
    """
    グラフ上の異常サインセット
    :param ax:subplot               サブプロット
    :param dt:                      日付軸
    :param dtlabel:                 異常有無
    :param y:list                   描画縦位置
    :param c:list                   色
    :return:
    """
    for k in range(len(dt)):
        if k > 0 and k < len(dt) - 1:
            if dtlabel[k - 1] == 0 and dtlabel[k] == 1 and dtlabel[k + 1] == 0:
                ax.text(dt[k], y, u"↓", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 0 and dtlabel[k] == 1 and dtlabel[k + 1] == 1:
                ax.text(dt[k], y, u"←", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 1 and dtlabel[k] == 1 and dtlabel[k + 1] == 0:
                ax.text(dt[k], y, u"→", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 1 and dtlabel[k] == 1 and dtlabel[k + 1] == 1:
                ax.text(dt[k], y, u"―", color=c, horizontalalignment="center")
        elif k == 0:
            if dtlabel[k] == 1 and dtlabel[k + 1] == 0:
                ax.text(dt[k], y, u"↓", color=c, horizontalalignment="center")
            elif dtlabel[k] == 1 and dtlabel[k + 1] == 1:
                ax.text(dt[k], y, u"←", color=c, horizontalalignment="center")
        elif k == len(dt) - 1:
            if dtlabel[k - 1] == 1 and dtlabel[k] == 1:
                ax.text(dt[k], y, u"→", color=c, horizontalalignment="center")
            elif dtlabel[k - 1] == 0 and dtlabel[k] == 1:
                ax.text(dt[k], y, u"↓", color=c, horizontalalignment="center")


# --------------------read ini file--------------------

def show_config(ini):
    """
    設定ファイルの全ての内容を表示する

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
    設定ファイルの特定のセクションの内容を表示する

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
    設定ファイルの特定セクションの特定のキー項目（プロパティ）の内容を表示する

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
    サブ画面：iniファイル選択初期画面の表示
    :param nwb:childObject
    :param parent:objct
    :return:
    """
    mbar = Menubar(nwb.root, nwb.on_close, parent.inifilename, menu_load, menu_save, dir_load=parent.default_inipath,
                   dir_save=parent.default_inipath, arg=parent)


def get_tabs(menu, filename):
    """
    サブ画面：設定情報読込みおよびグループソート
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
        messagebox.showerror(title=u"iniファイル読み込みエラー", message=u"エラー1：iniファイルにベンダー情報が記載されていません")
        return -1
    if int(parent.radio2.get()) == 0:
        if not cp.get("Vender", "Type") == "HUA":
            messagebox.showerror(title=u"iniファイル読み込みエラー", message=u"エラー２：ベンダーがＨＵＡに設定されています")
            return -1
    elif int(parent.radio2.get()) == 1:
        if not cp.get("Vender", "Type") == "ZTE":
            messagebox.showerror(title=u"iniファイル読み込みエラー", message=u"エラー３：ベンダーがＺＴＥに設定されています")
            return -1
    else:
        messagebox.showerror(title=u"iniファイル読み込みエラー", message=u"エラー４：iniファイルのベンダー情報はＨＵＡかＺＴＥに制限されています")
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
    サブ画面：iniファイル選択メニューバーの設定
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
    サブ画面：iniファイル選択saveコマンドの実行
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
    サブ画面：iniファイル選択初期ツリー表示
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
    サブ画面：iniファイル選択初期ツリー選択

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
    メイン画面コントロール配置

    txBox1＋openb1      :text+button     :入力ＫＰＩフォルダー＋選択
    txBox2＋openb2      :text+button     :出力グラフフォルダー＋選択
    rbtn2               :radio           :HUA or ZTE 選択
    nwb                 :button          :グラフ詳細選択
    rbtn4               :radio           :処理モード選択  u'モデリング＋グラフ', u'モデリング', u'グラフ'
    rbtn                :radio           :解析期間選択    u'3 Days', u'1 Week', u'Free'
    s_pane              :text            :開始日
    e_pane              :text            :終了日
    stbtn               :button          :スタートボタン
    pb1                 :progress        :プログレスバー
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

        # 入力ＫＰＩパターン
        # 蓄積データフォルダー
        self.strpath2 = parentdir + "/ref_KPI_Data"
        self.temppath = maindir + "/temp"
        self.backuppath = maindir + "/BackUp"
        self.learningpath = maindir.replace(chr(165), "/") + "/Learning_Data"
        # self.learningpath = 'C:/Users/wcp/Anaconda3/Scripts' + "/Learning_Data"
        self.default_inipath = parentdir + "\\ini"
        self.default_inipath_hua = self.default_inipath + "\\HUA_全選択.ini"
        self.default_inipath_zte = self.default_inipath + "\\ZTE_全選択.ini"

        self.strrpath = '.*(ZTE|HUA).*_.*.csv'
        # 蓄積データフォルダー
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

        # ラベル
        self.lblpane1 = LabelPane(self.toppane, u'KPIデータ選択', "white", C_BLUE)
        # テキストボックス
        self.txBox1 = ttk.Entry(self.lblpane1.botpane, width=20)
        self.txBox1.insert(tk.END, "")
        self.txBox1.pack(side=tk.LEFT, padx=5, expand=1, fill=tk.X)
        # ダイアログボックス
        self.openb = tk.Button(self.lblpane1.botpane, bg="white", fg=C_BLUE, relief='groove', text='Open',
                               command=self.opendir, width=10)
        self.openb.pack(side=tk.RIGHT, padx=5)
        self.lblpane1.pack(fill=tk.X, expand=1)
        self.txBox1.bind("<KeyRelease>", self.txbox1_text)

        # ラベル
        self.lblpane2 = LabelPane(self.toppane, u'グラフ出力先選択', "white", C_BLUE)
        # テキストボックス
        self.txBox2 = ttk.Entry(self.lblpane2.botpane, width=20)
        self.txBox2.insert(tk.END, "")
        self.txBox2.pack(side=tk.LEFT, padx=5, expand=1, fill=tk.X)
        # ダイアログボックス
        self.openb2 = tk.Button(self.lblpane2.botpane, bg="white", fg=C_BLUE, relief='groove', text='Open',
                                command=self.opendir2, width=10)
        self.openb2.pack(side=tk.RIGHT, padx=5)
        self.lblpane2.pack(fill=tk.X, expand=1)
        self.txBox2.bind("<KeyRelease>", self.txbox2_text)

        # ベンダー選択
        self.i1 = ImagePane(root, i1r)
        self.i1.pack(anchor=tk.W)
        self.midlblpane0 = LabelPane(self.i1.rpane, u'ベンダー', C_BLUE, "white")
        self.midlblpane0.pack()
        self.rbtn2 = []
        self.radio2.set(0)
        for i, x in enumerate((u'HUA', u'ZTE')):
            self.rbtn2.append(ttk.Radiobutton(self.midlblpane0.botpane, text=x, value=i, variable=self.radio2,
                                              state=tk.NORMAL, command=self.set_ini_default))
            self.rbtn2[i].pack(side=tk.LEFT)
        # 対象KPI選択
        self.i2 = ImagePane(root, i2r)
        self.i2.pack(anchor=tk.W)
        self.midlblpane1 = LabelPane(self.i2.rpane, u'対象KPI設定', C_BLUE, "white")
        self.midlblpane1.pack()
        self.nwb = NewWindowButton(self.midlblpane1.botpane, "選択", title="グラフ詳細選択", w=500, h=500, fg=C_BLUE, bg="white")
        self.inifilename = ""
        self.inifileshort = ""
        self.nwb.f_on_create = lambda: createnewwindow(self.nwb, self)
        self.nwb.b_create = lambda: [b.configure(state=tk.DISABLED) for b in self.rbtn2]
        self.nwb.b_close = lambda: [b.configure(state=tk.NORMAL) for b in self.rbtn2]
        self.nwb.pack(side=tk.LEFT)
        self.l1 = ttk.Label(self.midlblpane1.botpane, text=" ")
        self.l1.pack(side=tk.LEFT)

        # 処理モード選択
        self.i3 = ImagePane(root, i3r)
        self.i3.pack(anchor=tk.W)
        self.midlblpane2 = LabelPane(self.i3.rpane, u'処理モード', C_BLUE, "white")
        self.midlblpane2.pack()
        self.rbtn4 = []
        self.radio4.set(0)
        for i, x in enumerate((u'モデリング＋グラフ', u'モデリング', u'グラフ')):
            self.rbtn4.append(ttk.Radiobutton(self.midlblpane2.botpane, text=x, value=i, variable=self.radio4,
                                              state=tk.NORMAL))
            self.rbtn4[i].pack(side=tk.LEFT)

        # 解析期間選択
        self.i4 = ImagePane(root, i4r)
        self.i4.pack(anchor=tk.W)
        self.midlblpane3 = LabelPane(self.i4.rpane, u'解析期間', C_BLUE, "white")
        self.midlblpane3.pack(side=tk.LEFT)
        self.rbtn = []
        self.radio.set(0)
        for i, x in enumerate((u'3 Days', u'1 Week', u'Free')):
            self.rbtn.append(
                ttk.Radiobutton(self.midlblpane3.botpane, text=x, value=i, variable=self.radio, state=tk.NORMAL,
                                command=self.radio_state))
            self.rbtn[i].pack(side=tk.LEFT)

        # [開始日,終了日]をグループするウィンドウペイン
        self.datepane = ttk.PanedWindow(self.midlblpane3.botpane, orient=tk.VERTICAL, style="inv.TPanedwindow")
        self.datepane.pack(anchor=tk.W, expand=1)

        # テキストボックス
        self.s_pane = DatePane(self.datepane, u'　開始日', C_BLUE, "white", fg_enable=C_BLUE)
        self.e_pane = DatePane(self.datepane, u'　終了日', C_BLUE, "white", fg_enable=C_BLUE)
        self.radio_state()

        # スタートボタン
        self.startpane = ttk.PanedWindow(root)
        self.startpane.pack(expand=1)

        # ←ペイン
        self.startpane_l = ttk.PanedWindow(self.startpane)
        self.stbtn = tk.Button(self.startpane_l, text="START", state=tk.DISABLED, command=self.start_btn, bg="white",
                               fg=C_BLUE, relief='groove', height=2, width=12, highlightthickness=1,
                               highlightbackground=C_BLUE, highlightcolor=C_BLUE)
        self.stbtn.pack()
        self.startpane_l.pack(side=tk.LEFT, padx=5)

        # →ペイン
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

    # ベンダー選択によりデフォルト対象ＫＰＩファイル設定
    def set_ini_default(self):
        """
        ベンダー別デフォルトINIパス設定
        :return:
        """
        if int(self.radio2.get()) == 0:
            self.set_inifilepath(self.default_inipath_hua)
        else:
            self.set_inifilepath(self.default_inipath_zte)

    # 対象ＫＰＩファイル設定
    def set_inifilepath(self, filename):
        """
        デフォルトINIファイル読込み
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
            messagebox.showerror(title=u"iniファイル読み込みエラー", message=u"エラー５：iniファイル" + filename + u"が見つかりません")

    # メッセージの表示
    def message_dest(self):
        """
        デフォルトINIファイル読込み

        :return:
        """
        self.subWindow.destroy()

    def message_open(self):
        """
        メッセージの表示
        :return:
        """
        self.subWindow = tk.Toplevel()
        self.subWindow.title("メッセージ")
        sublabel = tk.Label(self.subWindow, text=self.msg)
        sublabel.pack()

    def txbox1_text(self, event):
        """
        KPI入力入力フォルダー選択のイベント

        :param event:event
        :return:
        """
        if self.txBox1.get() != "" and self.txBox2.get() != "":
            self.stbtn.configure(foreground="white", background=C_GOLD)
            self.stbtn_state(tk.NORMAL)
        else:
            self.stbtn.configure(foreground=C_BLUE, background="white")
            self.stbtn_state(tk.DISABLED)

    # テキストボックスのイベント
    def txbox2_text(self, event):
        """
        グラフ出力フォルダー選択のイベント

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
        KPI入力入力フォルダー選択のボタン

        :return:
        """
        dirname = tkfd.askdirectory(title=u'Kpiデータ選択')
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
        グラフ出力フォルダー選択のボタン

        :return:
        """
        dirname = tkfd.askdirectory(title=u'出力データ選択')
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
        KPI入力フォルダー選択のテキスト使用可否切り替え

        :param stat:int
        :return:
        """
        self.txBox1["state"] = stat

    def txBox2_state(self, stat):
        """
        グラフ出力フォルダー選択のテキスト使用可否切り替え

        :param stat:int
        :return:
        """
        self.txBox2["state"] = stat

    def radio_state(self):
        """
        グラフ出力期間選択ラジオボタン制御
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
        グラフ出力期間選択ラジオボタン制御
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
        ベンダー選択使用可否設定
        :param stat:int
        :return:
        """
        for rb in self.rbtn2:
            rb["state"] = stat

    def radio4_state2(self, stat):
        """
        出力モード選択使用可否
        :param stat:int
        :return:
        """
        for rb in self.rbtn4:
            rb["state"] = stat

    def stbtn_state(self, stat):
        """
        スタートボタン使用可否
        :param stat:
        :return:
        """
        self.stbtn["state"] = stat

    def stbtn_disable(self):
        """
        スタート:ボタン使用不可
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
        スタートボタン初期化
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
        スタート処理終了時
        :return:
        """
        self.stbtn_reset()
        if len(self.ijyoufiles) == 0:
            d = Dialogbox(self.master, "Finished", u"\n処理が完了しました\n\nKPI異常なし", image=i2r)
            d.centerwindow()
        else:
            d = Dialogbox(self.master, "Finished", u"\n処理が完了しました\n\nKPI異常あり\n出力フォルダを確認してください", image=i2r)
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
        進行状況の更新
        :return:
        """
        self.counter_done = sd["counter"]
        self.pb1.update_bar(int(100 * ((gui.donecount + self.counter_done) / (gui.taskcount + self.counter_task))))
        self.pb1.update_lbl(
            str(gui.donecount + self.counter_done) + "/" + str(gui.taskcount + self.counter_task) + " tasks completed")

    # cancel button pressed
    def on_cancel(self):
        """
        キャンセルボタン押下時
        :return:
        """
        self.pb1.update_lbl("Canceling...")
        self.stbtn.configure(state=tk.DISABLED, foreground=C_BLUE, background="white")
        sd["exit"] = True  # サブプロセス終了フラグ
        gui.thread_stop()  # スレッド終了フラグ

    def show_growth(self):
        """

        :return:
        """
        import objgraph
        objgraph.show_growth()

    def start_btn(self):
        """
        スタートボタン押下時
        :return:
        """
        try:
            if self.nwb.isopen():
                messagebox.showerror(title=u"KPI異常判定ツール エラー", message=u"エラー６：グラフ詳細選択を閉じてからスタートを押して下さい")
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
                messagebox.showerror(title=u"KPI異常判定ツール エラー", message=msg)
                self.stbtn_reset()
                return
            # 処理開始
            os.chdir(self.strpath2)
            self.counter_done = 0
            self.counter_task = 0
            sd["exit"] = False
            sd["counter"] = 0
            os.chdir(self.learningpath)
            for i, fullfile in enumerate(self.files):  # ＫＰＩデータ単位Loop
                gui.thread_start(lambda self=self, fullfile=fullfile, i=i: self.setpaths(fullfile, i),
                                 lambda self=self: self.updatepb())
                gui.thread_start(lambda self=self: self.fmtconv(),
                                 lambda self=self: self.updatepb())  # KPI+蓄積データの読み込みラベル付加
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
        入力チェック処理
        :return:
        """
        # 蓄積データホルダーに作業用ホルダー作成
        if os.path.exists(self.temppath) == False:
            os.mkdir(self.temppath)
        if os.path.exists(self.backuppath) == False:
            os.mkdir(self.backuppath)

        # 画面入力チェック
        self.pb1.update_lbl("checking data...")
        self.strpath = self.txBox1.get()
        self.strpath4 = self.txBox2.get()
        if os.path.isdir(self.strpath) == False:
            return u"エラー７：解析対象データフォルダ" + self.strpath + u"がありません"
        if os.path.isdir(self.strpath4) == False:
            return u"エラー８：グラフ出力先フォルダ" + self.strpath4 + u"がありません"
        if os.path.isdir(self.strpath2) == False:
            return u"エラー９：ＫＰＩ蓄積用フォルダ" + self.strpath2 + u"がありません"
        os.chdir(self.strpath)

        # 日付入力チェック
        if int(self.radio.get()) == 2:


            self.sdatetime = self.s_pane.getdatetime()
            self.edatetime = self.e_pane.getdatetime()

            if self.sdatetime is None:
                return u"エラー１８：開始日付エラー"
            if self.edatetime is None:
                return u"エラー１９：終了日付エラー"
            self.sdatetime = self.sdatetime.replace(hour=0)
            self.edatetime = self.edatetime.replace(hour=23)
            if self.sdatetime > self.edatetime:
                return u"エラー２０：開始日付>終了日付エラー"

        filesg = os.listdir(self.strpath)
        self.files = []
        # 対象ＫＰＩファイルセレクト
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
            return u"エラー１０：解析対象データフォルダ" + self.strpath + u"内に該当するファイルがありません"

        self.tmpfile = []
        self.ofile = []

        # 対象ＫＰＩ別チェック
        for k, file in enumerate(self.files):  # 解析対象データをループ
            # KPIデータ読み込み
            df = CSV.read(self.strpath + "/" + file, header=0, tempdir=self.temppath, tempfile="tmp_sjis.csv", nrows=1)
            is_empty = True
            for kpi in config2t.keys():
                if int(config2t[kpi]['KpiFlg01']['KpiFlg01']) == 1 and kpi in df.columns:
                    is_empty = False
            if is_empty:
                return u"エラー１１：解析対象データ" + self.strpath + "/" + file + u"内に該当するＫＰＩがありません"

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
                return u"エラー１２：解析対象データ" + self.strpath + "/" + file + u"にデータがありません"
            for i in range(2, len(test_data)):
                t1 = todatetime(test_data[i - 1][0])
                t2 = todatetime(test_data[i][0])
                # 重複データ確認
                wksa = (t2 - t1).seconds
                if wksa == 0:
                    return u"エラー１３：解析対象データ" + self.strpath + "/" + file + u"の" + test_data[i - 1][0] + u"が重複しています"
            # 蓄積データ判別
            vend = re.search('(ZTE|HUA)', file)
            if vend:
                vend.start
                # (ZTE|HUA)_xxxxx算出
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
                    return u"エラー１５：蓄積データフォルダ" + self.strpath2 + u"内に解析対象データと一致するファイルがありません"
                if fcnt > 1:
                    return u"エラー１６：蓄積データフォルダ" + self.strpath2 + u"に重複ファイルがあります"
            else:
                return u"エラー１４：解析対象データ" + file + u"のファイル名にHUA・ZTEが付いていません"

            # 蓄積データ読み込み
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
                    # 重複データ確認
                    wksa = (t2 - t1).seconds
                    if wksa == 0:
                        msg = u"エラー１７：蓄積データファイル" + self.strpath2 + "/" + self.ofile[k] + u"の" + train_data[i - 1][
                            0] + u"が重複しています"
                        return 1, msg

            # 作業用・蓄積ファイルの出力（重複データ削除）
            t1 = todatetime(test_data[1][0])
            if len(train_data) > 1:
                t2 = todatetime(train_data[-1][0])
            else:
                t2 = todatetime("2000-01-01 00:00:00")
            wksa = (t1 - t2).seconds
            tempfiles.add(self.temppath + "/tmp" + self.tmpfile[k])
            f = open(self.temppath + "/tmp" + self.tmpfile[k], 'w')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(train_data[0])  # ヘッダーを書き込む

            # 画面選択期間の妥当性チェック
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
                return u"エラー２２：蓄積データファイルにデータがありません"

            if int(self.radio.get()) == 2:
                if self.edatetime < todatetime(test_data[1][0]):
                    return u"エラー２１：最終日付<開始データ日付エラー"
                if self.sdatetime > todatetime(test_data[-1][0]):
                    return u"エラー２１：開始日付>最終データ日付エラー"
        return None

    def setpaths(self, fullfile, i):
        """
        関連フォルダーのチェックおよび作成
        :param fullfile:string
        :param i:int
        :return:
        """
        self.file = str(os.path.basename(fullfile))
        self.tmp = 'tmp' + self.tmpfile[i]
        self.bak = self.ofile[i]
        # グラフ保存ディレクトリー作成
        if int(self.radio2.get()) == 0:
            vend = re.search('(HUA)', self.file)
        else:
            vend = re.search('(ZTE)', self.file)
        # datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.strgpath = self.file[vend.start():-4]
        if os.path.exists(self.strpath4 + "/" + self.strgpath) == False:
            os.mkdir(self.strpath4 + "/" + self.strgpath)
        # モデル保存ディレクトリー作成
        if os.path.exists(self.learningpath + "/" + self.strgpath) == False:
            os.mkdir(self.learningpath + "/" + self.strgpath)
        dtnow = (str(pd.datetime.now())[0:-7]).replace(":", "-")  # yyyy-mm-dd hh:mm:ss.eeeeee to yyyy-mm-dd hh-mm-ss
        self.outpath = self.strpath4 + "/" + self.strgpath + "/" + dtnow
        os.mkdir(self.outpath)
        self.ijyoufile = self.outpath + "/" + self.strgpath + "_異常.csv"
        f = open(self.ijyoufile, 'w')
        f.write("KPI名,開始時刻,終了時刻,備考\n")
        f.close()

    def fmtconv(self):
        """
        KPI ＋ トレーニングデータの読込み　およびフォーマット調整

        :return:
        """
        # ＫＰＩデータ読み込み－－－＞numpy変換
        self.dfkpi = CSV.read(self.strpath + "/" + self.file, header=0, tempdir=self.temppath, tempfile="tmp_sjis.csv")
        if "Datetime" in self.dfkpi.columns:
            self.dfkpi.rename(columns={'Datetime': 'DateTime'}, inplace=True)
        self.dfkpi, _ = Time.df_fill(self.dfkpi)
        self.dfkpi['DateTime'] = pd.to_datetime(self.dfkpi['DateTime'])

        # 蓄積データ読み込み－－－＞numpy変換
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

        # 非KPIをカット
        cut = 0
        for lb in self.dftrain.columns[0:min(10, len(self.dftrain))]:
            if lb in NOT_KPI:
                cut = cut + 1

        # ラベル位置調整:トレーニングラベルを全て退避し削除
        wklbl = pd.DataFrame()
        for lb in self.dftrain.columns[cut:]:
            if lb[-6:] == "_label":
                wklbl[lb] = self.dftrain[lb]
                del self.dftrain[lb]
        for lb in self.dfkpi.columns[cut:]:
            if lb[-6:] == "_label":
                del self.dfkpi[lb]

        # ラベル位置調整:トレーニングラベルを改めて戻す
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
        KPI別メイン処理
        :return:
        """
        kpis = []
        for kpi in config2t.keys():
            if int(config2t[kpi]['KpiFlg01']['KpiFlg01']) == 1 and kpi in self.dfkpi.columns:
                kpis.append(kpi)
        self.counter_task = len(kpis) * len(self.files) * 3
        self.updatepb()
        # パス設定
        outpath = self.outpath
        ijyoupath = self.outpath + "/" + self.strgpath + "_異常"
        modelpath = self.learningpath + "/" + self.strgpath + "/"
        init_modelpath = self.learningpath + "/init.h5"
        # フラグ設定
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
            title = "(Free)" + (self.s_pane.getdatestr(sep="/") + "～" + self.e_pane.getdatestr(sep="/"))
        # n = mp.cpu_count()-1
        n = procs
        if n <= 1:  # プロセス数 == 1
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
            # args・filepaths作成
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
        new蓄積データ出力
        :return:
        """
        # トレーニング、Kpiデータ連結
        # 元のトレーニングデータをバックアップ
        shutil.move(self.strpath2 + "/" + self.bak, self.backuppath + "/" + self.bak)
        # 最終のトレーニング日時をファイル名に付加し保存
        tstamp = self.dfkpi['DateTime'].iloc[-1].strftime("%Y-%m-%d-%H")
        CSV.write(self.dfkpi, self.strpath2 + "/" + tstamp + "_" + self.strgpath + ".csv")


if __name__ == '__main__':
    mp.freeze_support()
    sd = Pooler.shared_dict()
    argc = len(sys.argv)
    root = tix.Tk()
    root.title('KPI異常判定ツール')
    app = Application(master=root)
    # debug mode
    sd["debug"] = False
    if sd["debug"]:
        app.radio.set(1)  # 1 week
        app.radio2.set(0)  # HUA
        # app.radio2.set(1)  #ZTE
        app.set_inifilepath("C:/Users/wcp/Desktop/KPI異常検知ツール_ win64bit_V1.7/ini/HUA_全選択.ini")
        app.txBox1.insert(0, "C:/Users/wcp/Desktop/KPI異常検知ツール_ win64bit_V1.7/test/yoko_h1")
        app.txBox2.insert(0, "C:/Users/wcp/Desktop/KPI異常検知ツール_ win64bit_V1.7/test/yoko_h1")
        app.stbtn.configure(text="START", foreground="white", background=C_GOLD, command=app.start_btn, state="normal")
    gui = ThreadedGUI(root, f_stop=lambda app=app: app.stbtn_reset())
    app.mainloop()
