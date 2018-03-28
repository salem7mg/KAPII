
from aux_misc import DateTimeReader

import os
import shutil
import time
import numpy as np
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Mplt():
    figure = None

    def _fill(ax,x,y,c,alpha=1.0,vlines=False):
        if vlines:
            ax.vlines(x,0,y,color=c,alpha=alpha)
            return
        if isinstance(c, (list, tuple, np.ndarray)):
            for i in range(len(x)):
                if y[i] != 0:
                    ax.fill_between([x[i],x[i]+1],0,[y[i],y[i]],facecolor=c[i],alpha=alpha)
        else:
            for i in range(len(x)):
                if y[i] != 0:
                    ax.fill_between([x[i],x[i]+1],0,[y[i],y[i]],facecolor=c,alpha=alpha)

    #plot fill from 0 to y
    #y: [y1,y2,y3,...]
    #c: [c1,c2,c3,...] or single color
    def plotfill(y,c,alpha=1.0,vlines=False,title=None):
        if Mplt.figure == None:
            Mplt.figure = plt.figure()
        ax = Mplt.figure.add_subplot(111)
        #title
        if title:
            plt.title(title)
        #plot
        x = np.arange(len(y))
        Mplt._fill(ax,x,y,c,alpha,vlines)
        return ax

    #plot multiple arrays in the same window
    #x: [[xlist1],...] or [n1,n2,n3,...]
    #y: [[ylist1],...]
    def plots(x,y,c=None,title=None,fill=False,sharex=True,sharey=True):
        n = len(y)
        fig, axs = plt.subplots(n, sharex=sharex, sharey=sharey)
        #title
        if title:
            plt.title(title)
        #plot
        fig.subplots_adjust(hspace=0)
        if isinstance(x[0], (list, tuple, np.ndarray)):
            if fill:
                for i in range(n):
                    Mplt._fill(axs[i],x[i],y[i],c[i])
            else:
                for i in range(n):
                    axs[i].plot(x[i],y[i],".")
        else:
            if fill:
                for i in range(n):
                    Mplt._fill(axs[i],x,y[i],c[i])
            else:
                for i in range(n):
                    axs[i].plot(x,y[i],".")
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    #format: if True, format t to datetime
    def plotdatetime(t,y,title=None,format=True):
        #times = pd.date_range('2000/01/01', periods=100, freq='60min')
        if format:
            times = [DateTimeReader.todatetime(str) for str in t]
        if Mplt.figure == None:
            Mplt.figure = plt.figure()
        ax = Mplt.figure.add_subplot(111)
        #title
        if title:
            plt.title(title)
        #plot
        plt.plot(times, y)
        #axis
        plt.xlim(xmin=times[0],xmax=times[-1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))  #set x axis label format
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO))
        #plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        #plt.gca().xaxis.set_major_locator(mdates.HourLocator())              #set x axis ticks per hour
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=10)              #label rotation and font size
        plt.tight_layout()                                                    #make room for label
        #Mplt.figure.autofmt_xdate()
        return ax

    #x: data (format:[[x1,y1],[x2,y2]])
    #title: title
    #a: annotations
    #c: colors
    def plot2d(x,title=None,a=[],c=[],alpha=1.0):
        xs = [e[0] for e in x]
        ys = [e[1] for e in x]
        if Mplt.figure == None:
            Mplt.figure = plt.figure()
        ax = Mplt.figure.add_subplot(111)
        #title
        if title:
            plt.title(title)
        #plot
        if len(c) == 0:  #no color
            ax.scatter(xs,ys,alpha=alpha)
        else:
            ax.scatter(xs,ys,c=c,alpha=alpha)
        #annotations
        for i, txt in enumerate(a):
            ax.text(xs[i],ys[i],txt)
        #axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return ax

    #x: data (format:[[x1,y1,z1],[x2,y2,z2]])
    #title: title
    #a: annotations
    #c: colors
    def plot3d(x,title=None,a=[],c=[],alpha=1.0):
        xs = [e[0] for e in x]
        ys = [e[1] for e in x]
        zs = [e[2] for e in x]
        if Mplt.figure == None:
            Mplt.figure = plt.figure()
        ax = Mplt.figure.add_subplot(111, projection='3d')
        #title
        if title:
            plt.title(title)
        #plot
        if len(c) == 0:  #color gradient
            cs = zs
            #norm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
            norm = matplotlib.colors.Normalize(vmin=np.percentile(cs, 10), vmax=np.percentile(cs, 90))
            cmap = plt.get_cmap('jet')
            scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
            scalarMap.set_array(cs)
            c = scalarMap.to_rgba(cs)
            ax.scatter(xs,ys,zs,c=c,alpha=alpha)
            Mplt.figure.colorbar(scalarMap)
        else:
            ax.scatter(xs,ys,zs,c=c,alpha=alpha)
        #annotations
        for i, txt in enumerate(a):
            ax.text(xs[i],ys[i],zs[i],txt)
        #axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax

    def plot2d_cluster(ax,tsne_x,dbscan_x):
        if not isinstance(tsne_x,np.ndarray):
            tsne_x = np.array(tsne_x)
        if not isinstance(dbscan_x,np.ndarray):
            dbscan_x = np.array(dbscan_x)
        min_dbscan = np.min(dbscan_x)
        max_dbscan = np.max(dbscan_x)
        color=cm.brg(np.linspace(0,1,max_dbscan - min_dbscan+1))
        for i in range(min_dbscan, max_dbscan+1):
            if i == -1:
                c = "black"
            else:
                c = color[i-min_dbscan]
            temp = tsne_x[dbscan_x == i]  #tsne_x[j] where dbscan_x[j] == i
            ax.scatter(temp[:,0], temp[:,1], s=10, c=c, marker=".")
            ax.text(temp[:,0][0], temp[:,1][0], str(i), color="black", size=16)

    def plot3d_cluster(ax,tsne_x,dbscan_x):
        if not isinstance(tsne_x,np.ndarray):
            tsne_x = np.array(tsne_x)
        if not isinstance(dbscan_x,np.ndarray):
            dbscan_x = np.array(dbscan_x)
        min_dbscan = np.min(dbscan_x)
        max_dbscan = np.max(dbscan_x)
        color=cm.brg(np.linspace(0,1,max_dbscan - min_dbscan+1))
        for i in range(min_dbscan, max_dbscan+1):
            if i == -1:
                c = "black"
            else:
                c = color[i-min_dbscan]
            temp = tsne_x[dbscan_x == i]  #tsne_x[j] where dbscan_x[j] == i
            ax.scatter(temp[:,0], temp[:,1], temp[:,2], s=10, c=c, marker=".")
            ax.text(temp[:,0][0], temp[:,1][0], temp[:,2][0], str(i), color="black", size=16)

    def resize(w=600,h=600,x=20,y=20):
        cfm = plt.get_current_fig_manager()
        cfm.window.wm_geometry(str(w)+"x"+str(h)+"+"+str(x)+"+"+str(y))

    def show():
        plt.show()

    def close():
        plt.close()
        Mplt.figure = None

    def savefig(path):
        plt.savefig(path)
        plt.clf()

class Mpltfont():
    #install ttf file
    def install(ttf_path="C:/ttf/ipaexg.ttf"):
        ttf_file = os.path.basename(ttf_path)
        font_path = "C:/Windows/Fonts"
        if not os.path.exists(ttf_path):
            return -1
        if os.path.exists(font_path):
            if not os.path.exists(font_path + ttf_file):
                shutil.copy(ttf_path, font_path + ttf_file)
            return 1

    #set default font
    def set(fontname="IPAexGothic"):
        if not Mpltfont._setfont(fontname):
            Mpltfont._delcache()  #delete font cache
            time.sleep(2)
            Mpltfont._rebuild()   #rebuild font cache
            time.sleep(2)
            Mpltfont._setfont(fontname)
            time.sleep(2)

    #delete cache
    def _delcache():
        p1 = matplotlib.get_configdir() + "/fontList.py3k.cache"
        p2 = matplotlib.get_cachedir() + "/fontList.py3k.cache"
        p3 = matplotlib.get_configdir() + "/fontList.json"
        p4 = matplotlib.get_cachedir() + "/fontList.json"
        for p in [p1,p2,p3,p4]:
            if os.path.isfile(p):
                os.remove(p)
                print("deleted matplotlib font cache:",p)
            else:
                print("path not found:",p)

    #rebuild cache
    def _rebuild():
        print("rebuilding matplotlib font cache...")
        matplotlib.font_manager._rebuild()

    #set plot font
    def _setfont(fontname):
        if fontname in Mpltfont.fontnames():
            matplotlib.rc('font',**{'family':fontname})
            #print("matplotlib font set:",fontname)
            return True
        else:
            #print("matplotlib font not found:",fontname)
            return False

    #get font
    def getfont():
        return matplotlib.rcParams['font.family'][0]

    def systemfonts():
        return matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    def fontnames():
        return [f.name for f in matplotlib.font_manager.fontManager.ttflist]

    def test():
        print(Mpltfont.systemfonts())
        print(Mpltfont.fontnames())
        print("default font:",Mpltfont.getfont())
        Mpltfont.install("C:/ttf/ipaexg.ttf")
        Mpltfont.set('IPAexGothic')
        print("default font:",Mpltfont.getfont())

