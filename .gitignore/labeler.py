
from aux_math import MatrixProfile,Outlier,RollingWindow,Scaler
from aux_misc import Filemngr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

import math
import numpy as np

def labelit(t,y,period=24*7,title="",savepath=None,plot=False):
    if savepath == None:
        savepath1 = None
        savepath2 = None
    else:
        savepath1 = savepath + "/" + title + "_matrix.png"
        savepath2 = savepath + "/" + title + "_profile.png"
        Filemngr.mkdir(savepath)

    #compute decomposition
    y_scl = Scaler.scale(y)
    y_trend,y_detrend,y_seasonal,y_irregular = RollingWindow.decompose(y_scl,trend_interval=period*2,trend_model="add",trend_center="mean",seasonal_interval=period,seasonal_model="add",seasonal_center="median")

    out = Outlier.std(y_irregular,5)  #5 standard deviations
    y_norm = Outlier.replace(y,out,[y[i]-y_irregular[i] for i in range(len(y))])
    y_nonzero = Scaler.scale(y_norm)

    #compute matrix profile
    ad_cmp = ["mean","var","lrdiff"]
    ad_elb = [5,5,10,10]
    ad_mat = [[] for _ in range(len(ad_cmp))]
    ad_sum = [[] for _ in range(len(ad_cmp))]
    ad_thr = [[] for _ in range(len(ad_cmp))]
    ad_flg = [[] for _ in range(len(ad_cmp))]

    for i in range(len(ad_cmp)):
        if plot:
            ad_mat[i],_,ad_sum[i],ad_thr[i],ad_flg[i] = MatrixProfile.extract(y_nonzero,period+1,elbow_weight=ad_elb[i],matrix_type=ad_cmp[i],profile_type="sum",return_type="mph")
        else:
            _,ad_sum[i],ad_thr[i],ad_flg[i] = MatrixProfile.extract(y_nonzero,period+1,elbow_weight=ad_elb[i],matrix_type=ad_cmp[i],profile_type="sum",return_type="ph")
    #plot profile
    if plot:
        MatrixProfile.plot(title,ad_cmp,ad_mat,savepath=savepath1,show=False)
        n = 9
        fig, axes = plt.subplots(n,sharex=True,figsize=(10,6))
        for i in range(0,n):
            axes[i].set_yticks([])
            axes[i].yaxis.set_label_coords(-0.05,0.5)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
            axes[i].grid(True)
        fig.subplots_adjust(hspace=0)
        axes[0].set_ylabel("trend", rotation=0, size='large')
        axes[0].plot(t,y)
        axes[0].plot(t,y_trend)
        axes[1].set_ylabel("seasonal", rotation=0, size='large')
        axes[1].plot(t,y_detrend)
        axes[1].plot(t,y_seasonal)
        axes[2].set_ylabel("noise", rotation=0, size='large')
        axes[2].plot(t,y_irregular)
        titles = ["avg_sum","var_sum","dif_sum"]
        colors = ["yellow","lightblue","pink"]
        for i in range(len(ad_sum)):
            axes[i+3].set_ylabel(titles[i], rotation=0, size='large', color=colors[i])
            axes[i+3].plot(t,[ad_thr[i]]*len(t),color="red")
            axes[i+3].plot(t,ad_sum[i])
            axes[i+3].plot(t,sorted(ad_sum[i]),color="green")
        #outlier
        axes[7].set_ylabel("out", rotation=0, size='large')
        axes[7].plot(t,out)
        axes[8].set_ylabel("norm", rotation=0, size='large')
        axes[8].set_ylim(min(y_norm),max(y_norm))
        axes[8].plot(t,y_norm)
        axes[8].vlines(t,0,[0]*len(y_nonzero),color="gray",alpha=0.5)
        for i in range(len(ad_sum)):
            axes[8].vlines(t,0,ad_flg[i],color=colors[i],alpha=0.5)
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO))
        plt.setp(axes[8].get_xticklabels(), rotation=45, fontsize=7, ha='right')
        plt.suptitle(title)
        #plt.show()
        if savepath2 != None:
            plt.savefig(savepath2)
        plt.close()
    labels = [ad_flg[0][i] or ad_flg[1][i] or ad_flg[2][i] for i in range(len(ad_flg[0]))]
    out = [int(o == -1) for o in out]
    labels = [labels[i] or out[i] for i in range(len(labels))]
    return labels
