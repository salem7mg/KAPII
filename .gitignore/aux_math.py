"""Auxiliary math functions.
Activation           #activation function (https://en.wikipedia.org/wiki/Activation_function)
Arr                  #[GENERAL]array
Cluster              #tsne dbscan clustering
ConvexHull           #convex hull (https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain)
Convolution          #perform 1d convolution using fft (https://en.wikipedia.org/wiki/Convolution)
CUSUM                #CUSUM control chart for anomaly detection (http://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)
Dispersion           #[STATISTICS]measures of dispersion
Elbow                #elbow detection
Entropy              #entropy
Fc                   #math functions
Fscore               #fscore (https://en.wikipedia.org/wiki/F1_score)
KDE                  #kernel density estimation (https://en.wikipedia.org/wiki/Kernel_density_estimation)
Kernel               #kernels for convolution (https://en.wikipedia.org/wiki/Window_function) and PMF (https://en.wikipedia.org/wiki/Probability_mass_function)
Local                #local min max search
Mat                  #[GENERAL]matrix
MatrixProfile        #matrix profile
Metrics              #metrics
Optics               #[OPTICS]clustering (https://en.wikipedia.org/wiki/OPTICS_algorithm) (https://en.wikipedia.org/wiki/Centroid)
Outlier              #[PREPROCESS]process outliers in data
Partition            #partition values into bins
PCA                  #principal component analysis (https://en.wikipedia.org/wiki/Principal_component_analysis)
Point                #[OPTICS]point data
Points               #create (x,y) point data
Proximate            #check proximity of two values
RollingWindow        #rolling window
Scaler               #[PREPROCESS]scale data (http://scikit-learn.org/stable/modules/preprocessing.html)
Shape                #[STATISTICS]measures of shape
Spectrum             #fft period detection
Time                 #time partitioner
TimePartition        #time partition object
Trough               #[OPTICS]trough detection
Vec                  #[GENERAL]vector
"""

from aux_misc import DateTimeReader
from aux_misc import PriorityQueue
from aux_plot import Mplt

import bisect
import datetime
import functools
import math
import operator
import random
import time
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict  #keep insert order

"""
import scipy.spatial as spatial
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE   #project higher dimensions to lower dimensions
from sklearn.cluster import DBSCAN  #clustering algorithm
"""

class Activation():
    def sigmoid(x):
        for i in range(len(x)):
            x[i] = 1/(1+math.exp(-x[i]))
        return x

    def tanh(x):
        for i in range(len(x)):
            a = math.exp(x[i])
            b = math.exp(-x[i])
            x[i] = (a-b)/(a+b)
        return x

class Arr():
    #return the area under the curve
    #w: width
    def area(x,w):
        if len(x) <= 1:
            return 0
        elif len(x) == 2:
            e = (x[0]+x[-1])/2
            return e*w
        else:
            e = (x[0]+x[-1])/2
            return (sum(x[1:-1])+e)*w

    #bin values
    #example:
    # bin([1,2,3],0,10,10) = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    # bin([1,2,3],0,3,10) = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    def bin(x,thr_min,thr_max,n=5):
        r = [0]*n
        dx = (thr_max-thr_min)/n  #bucket size
        if dx == 0:
            mid = int((n-1)/2)
            r[mid] = len(x)
        else:
            for i in range(len(x)):
                index = int((x[i]-thr_min)/dx)
                if index > n-1:
                    index = n-1
                r[index] = r[index] + 1
        return r

    #shift 1s to right or left
    #example:
    # binary_shift([0,0,1,0,0],1,"right") = [0,0,0,1,0,0]
    # binary_shift([1,0,0,0,1],1,"right") = [1,1,0,0,0,1]
    # binary_shift([0,0,1,0,0],1,"left")  = [0,1,0,0,0,0]
    # binary_shift([1,0,0,0,1],1,"left")  = [1,0,0,0,1,1]
    def binary_shift(x,n,type="right"):
        if type == "right":
            if x[0]:
                return [1]*n + list(x)
            else:
                return [0]*n + list(x)
        elif type == "left":
            if x[-1]:
                return list(x) + [1]*n
            else:
                return list(x) + [0]*n

    #blur 1s to right or left
    #example:
    # binary_blur([0,0,1,0,0],1,"right") = [0,0,1,1,0,0]
    # binary_blur([1,0,0,0,1],1,"right") = [1,1,0,0,1,1]
    # binary_blur([0,0,1,0,0],1,"left")  = [0,1,1,0,0,0]
    # binary_blur([1,0,0,0,1],1,"left")  = [1,0,0,1,1,0]
    def binary_blur(x,n,type="right"):
        x = list(x) + [0]*n
        if type == "right":
            for i in range(len(x)-1,-1,-1):
                if x[i]:
                    if i+1+n > len(x):
                        stop = len(x)
                    else:
                        stop = i+1+n
                    for j in range(i+1,stop,1):
                        if x[j]:
                            break
                        else:
                           x[j] = 1
        elif type == "left":
            for i in range(0,len(x),1):
                if x[i]:
                    if i-1-n < 0:
                        stop = -1
                    else:
                        stop = i-1-n
                    for j in range(i-1,stop,-1):
                        if x[j]:
                            break
                        else:
                           x[j] = 1
        return x

    #clip values less than minvalue to minvalue
    #example:
    # clip([1,2,3,-1],0) = [1,2,3,0]
    def clip(x, minvalue=0):
        r = [0]*len(x)
        for i in range(len(x)):
            if x[i] < minvalue:
                r[i] = minvalue
            else:
                r[i] = x[i]
        return r

    #compress array using run length encoding
    #example:
    # compress([1,1,1,1,2,2,2,3,3]) = [[1,4],[2,3],[3,2]]
    def compress(x):
        r = []
        curr = x[0]
        count = 1
        for i in range(1,len(x)):
            if x[i] == curr:
                count = count + 1
            else:
               r.append([x[i-1],count])
               curr = x[i]
               count = 1
        r.append([x[-1],count])
        return r

    #dice list
    #example:
    # dice([1,2,3,4,5]) = [1,3,5]
    def dice(x):
        r = []
        for i in range(0,len(x),2):
            r.append(x[i])
        return r

    #create a new array that is n times larger
    #copy from array each element to sth position in new array
    #example:
    # expand([1,2,3],2,0,0) = [1,0,2,0,3,0]
    # expand([1,2,3],2,1,0) = [0,1,0,2,0,3]
    # expand([1,2,3],2,1,9) = [9,1,9,2,9,3]
    def expand(x,n,s,v):
        s = s % n
        r = [v]*len(x)*n
        for i in range(len(x)):
            r[i*n+s] = x[i]
        return r

    #find i where x[i] = v
    #return -1 if not found
    #example:
    # find([1,2,3,2,1],2,False) = 1
    # find([1,2,3,2,1],2,True) = 3
    def find(x,v,reverse=False):
        if reverse:
            for i in range(len(x)-1,-1,-1):
                if x[i] == v:
                    return i
        else:
            for i in range(0,len(x),1):
                if x[i] == v:
                    return i
        return -1

    #find range in x where x[i]>=a and x[i]<=b
    #example:
    # findrange([1,2,3,4,5],6,0) = (-1,-1)
    # findrange([1,2,3,4,5],0,6) = (0,4)
    # findrange([1,2,3,4,5],2,4) = (1,3)
    # findrange([1,2,3,4,5],3,3) = (2,2)
    # findrange([1,2,3,4,5],2.9,3.1) = (2,2)
    def findrange(x,a,b):
        i1 = -1
        i2 = -1
        i = 0
        while i < len(x):
            if x[i] >= a:
                i1 = i
                break
            i = i + 1
        i = len(x)-1
        while i > -1:
            if x[i] <= b:
                i2 = i
                break
            i = i - 1
        return i1,i2

    #flatten list of lists by one floor
    #example:
    # flatten([1,2,[3,4],5]) = [1,2,3,4,5]
    # flatten([1,2,[3,[4]],5]) = [1,2,3,[4],5]
    def flatten(x):
        r = []
        if isinstance(x, (list, tuple, np.ndarray)):
            for l in x:
                if isinstance(l, (list, tuple, np.ndarray)):
                    for e in l:
                        r.append(e)
                else:
                    r.append(l)
        return r

    #get the longest interval that starts with start and ends with end
    #example:
    # index_ends([3,0,1,2,3,0,1,2],0,3) = 1,4
    def index_ends(x,start=0,end=23):
        s = Arr.find(x,start,reverse=False)
        if s == -1:
            return None
        e = Arr.find(x,end,reverse=True)
        if e == -1:
            return None
        if s >= e:
            return None
        return s, e

    #wrap (reuse values) near end points of array
    #example:
    # index_wrap([3,0,1,2,3,0,1,2],["a","b","c","d","e","f","g","h"],0,3) = [0,1,2,3,0,1,2,3,0,1,2,3],['b','c','d','a','b','c','d','e','f','g','h','e']
    #before:
    #a,b,c,d,e,f,g,h
    #3,0,1,2,3,0,1,2
    #  ^st   ^ed
    #after:
    #      a,b,c,d,e,f,g,h
    #b,c,d,a,b,c,d,e,f,g,h,e
    #0,1,2,3,0,1,2,3,0,1,2,3
    #        ^st   ^ed
    def index_wrap(x,y,start=0,end=23):
        st,ed = Arr.index_ends(x,start,end)
        mx = x[st:ed+1]
        my = y[st:ed+1]
        if st > 0:
            edx = x[0:st]
            lx = x[st:st+end-len(edx)+1] + edx
            edy = y[0:st]
            ly = y[st:st+end-len(edy)+1] + edy
        else:
            lx = []
            ly = []
        if ed < len(x)-1:
            stx = x[ed+1:len(x)]
            rx = stx + x[ed-end+len(stx):ed+1]
            sty = y[ed+1:len(y)]
            ry = sty + y[ed-end+len(sty):ed+1]
        else:
            rx = []
            ry = []
        return lx+mx+rx,ly+my+ry

    #interlace two arrays
    #example:
    # interlace([1,2,3],[4,5]) = [1, 4, 2, 5, 3]
    # interlace([1,2,3],[4,5,6]) = [1, 4, 2, 5, 3, 6]
    def interlace(x,y):
        r = [None]*(len(x)+len(y))
        for i in range(0,len(x)):
            r[i*2] = x[i]
        for i in range(0,len(y)):
            r[i*2+1] = y[i]
        return r

    #interpolation
    #n: number of values to interpolate between each interval
    #f: interpolation function
    #example:
    # interpolate([1,4,9],1,Arr.inter_linear)     = [1.0, 2.50, 4.0, 6.50, 9.0]
    # interpolate([1,4,9],1,Arr.inter_polynomial) = [1.0, 2.25, 4.0, 6.25, 9.0]
    # interpolate([1,4,9],1,Arr.inter_qspline)    = [1.0, 2.25, 4.0, 6.25, 9.0]
    def interpolate(x,n=1,f=None):
        if f == None:
            f = Arr.inter_linear
        r = []
        for i in range(len(x)-1):
            r.append(x[i])
            for j in range(1,n+1):
                k = j/(n+1)
                r.append(f(x,i+k))
        r.append(x[i+1])
        return r

    #linear interpolation
    #x: array
    #k: index
    #example:
    # inter_linear([1,4,9],0.5) = 2.50
    def inter_linear(x,k):
        i = int(k)
        r = k-i
        if r == 0:
            return x[i]
        else:
            return x[i] + (x[i+1]-x[i])*r

    #lagrange polynomial interpolation (https://en.wikipedia.org/wiki/Polynomial_interpolation)
    #x: array
    #k: index
    #example:
    # inter_polynomial([1,4,9],0.5) = 2.25
    #p(x) = [(x-x1)(x-x2)...(x-xn)]/[(x0-x1)(x0-x2)...(x0-xn)]*y0+
    #       [(x-x0)(x-x2)...(x-xn)]/[(x1-x0)(x1-x2)...(x1-xn)]*y1+...
    #       [(x-x0)(x-x2)...(x-x'n-1')]/[(xn-x0)(xn-x1)...(xn-x'n-1')]*yn
    #     = SUM[i=0 to n:MUL[j=0 to n,i!=j:(x-xj)/(xi-xj)]*yi]
    #warning: even if inputs are all positive, output may contain negative values
    def inter_polynomial(x,k):
        sum = 0
        for i in range(len(x)):
            mul1 = 1
            mul2 = 1
            for j in range(len(x)):
                if i != j:
                    mul1 = mul1*(k-j)
                    mul2 = mul2*(i-j)
            sum = sum + mul1/mul2*x[i]
        return sum

    #quadratic interpolation using spline
    #x: array
    #k: index
    #example:
    # inter_qspline([1,4,9],0.5) = 2.25
    #warning: even if inputs are all positive, output may contain negative values
    def inter_qspline(x,k):
        i = int(k)
        r = k-i
        if r == 0:
            return x[i]
        elif i == 0:
            lol = Arr.inter_polynomial(x[0:3],0.0+r)        #left of left
            return lol
        elif i == len(x)-2:
            ror = Arr.inter_polynomial(x[-3:len(x)],1.0+r)  #right of right
            return ror
        else:
            rol = Arr.inter_polynomial(x[i-1:i+2],1.0+r)    #right of left
            lor = Arr.inter_polynomial(x[i:i+3],0.0+r)      #left of right
            return (rol+lor)/2

    #divide array into left, center, right symmetric sub arrays
    #len(left+center+right) == len(y)
    # lcr([0,1,2,3,4],0) = [], [[0], [1], [2], [3], [4]], []
    # lcr([0,1,2,3,4],1) = lcr([0,1,2,3,4],0)
    # lcr([0,1,2,3,4],2) = [[0, 1]], [[0, 1, 2], [1, 2, 3], [2, 3, 4]], [[3, 4]]
    # lcr([0,1,2,3,4],3) = lcr([0,1,2,3,4],2)
    # lcr([0,1,2,3,4],4) = [[0, 1, 2], [0, 1, 2, 3]], [[0, 1, 2, 3, 4]], [[1, 2, 3, 4], [2, 3, 4]]
    # lcr([0,1,2,3,4],5) = lcr([0,1,2,3,4],4)
    def lcr(x,interval):
        odd = interval-interval%2+1  #odd interval size
        row = int(interval/2)
        col = int(interval/2)+1
        left = [x[0:i+col] for i in range(0,row)]
        center = [x[i:i+odd] for i in range(0,len(x)-odd+1)]
        right = [x[len(x)-i-col:len(x)] for i in range(row-1,-1,-1)]
        return left,center,right

    #fill array to length n with 0
    #example:
    # lenfill([1,2,3],4) = [1,2,3,0]
    def lenfill(x,n):
        i = len(x)
        while i < n:
            x = np.append(x,0)
            i = i + 1
        return x

    #return a forward index map
    #example:
    # map_forward([2,3,0,1,2,3,0,1,2,3,0]) = [2,3,0,1]
    def map_forward(x):
        for i in range(1,len(x),1):
            if x[0] == x[i]+1:
                return x[0:i+1]
        return None

    #return a backward index map
    #example:
    # map_backward([2,3,0,1,2,3,0,1,2,3,0]) = [1,2,3,0]
    def map_backward(x):
        for i in range(len(x)-2,-1,-1):
            if x[-1] == x[i]-1:
                return x[i:len(x)]
        return None

    #match array length to smallest length of x,y
    #example:
    # matchlen([1],[2,3]) = [1],[2]
    def matchlen(x,y):
        if len(x) < len(y):
            return x,y[:len(x)]
        else:
            return x[:len(y)],y

    #find max within range
    def maxr(x,xi,xf):
        vmax = x[xi]
        for i in range(xi,xf):
            if x[i] > vmax:
                vmax = x[i]
        return vmax

    #find min within range
    def minr(x,xi,xf):
        vmin = x[xi]
        for i in range(xi,xf):
            if x[i] < vmin:
                vmin = x[i]
        return vmin

    #fill nan values
    def nan_fill(x,fill="mean"):
        if fill == None:
            return x
        elif fill == "mean":
            intervals = Arr.nan_intervals(x)
            for iv in intervals:
                l = iv[0]-1
                r = iv[1]+1
                if l == -1 and r == len(x):  #all empty
                    return x
                elif l == -1:
                    for i in range(l+1,r):
                        x[i] = x[r]
                elif r == len(x):
                    for i in range(l+1,r):
                        x[i] = x[l]
                else:
                    m = (x[l]+x[r])/2
                    for i in range(l+1,r):
                        x[i] = m
        elif isinstance(fill, (int, float)):
            for i in range(len(x)):
                if math.isnan(x[i]):
                    x[i] = fill
        return x

    #return nan intervals
    def nan_intervals(x):
        i = 0
        l = 0
        r = 0
        intervals = []
        while i < len(x):
            if not math.isnan(x[i]):
                while not math.isnan(x[i]):  #skip until non-number (find l)
                    i = i + 1
                    if i >= len(x):
                        break
                l = i
            else:
                while math.isnan(x[i]):      #skip until number (find r)
                    i = i + 1
                    if i >= len(x):
                        break
                r = i-1
                intervals.append([l,r])
        return intervals

    #partition array into parts of n elements
    #example:
    # partition([1,2,3,4],2,0,1) = [[1,2],[3,4]]
    # partition([1,2,3,4],2,0,0) = [[1],[3]]
    # partition([1,2,3,4],2,1,1) = [[2],[4]]
    def partition(x,n=24,ci=None,cf=None,cut=True):
        if ci == None:
            ci = 0
        if cf == None:
            cf = n-1
        r = []
        if cut:
            l = len(x) - (len(x) % n)
            for i in range(0, l, n):
                r.append(x[i:i + n][ci:cf+1])
        else:
            l = len(x)
            for i in range(0, l, n):
                r.append(x[i:i + n][:])
        return r

    #one pass partition with spanning
    #labels: labels list
    #v: allowed value in label
    #n: partition size
    #s: span
    #return partition start indices, span length
    def partition_span(labels,v,n,s):
        r = []
        d = {}
        span = set()
        head = n-1
        tail = 0
        max_len = 0
        for i in range(0,n):
            if labels[i] != v:
                head = i
                break
        while head < len(labels):
            if labels[head] != v:
                tail = head + 1
            elif head-tail == n-1:
                key = tail%s
                r.append(tail)
                span.add(key)
                if key in d:
                    d[key] = d[key] + [tail]
                else:
                    d[key] = [tail]
                tail = tail + 1
            head = head + 1
        rd = []
        if s == len(span):
            for key in d.keys():
                l = len(d[key])
                if l > max_len:
                    max_len = l
            for key in d.keys():
                l = len(d[key])
                for i in range(max_len):
                    rd.append(d[key][i%l])
        return r,rd,len(span)

    #partition with one shift
    #x: list
    #n: partition size
    #e: endpoint
    #example:
    # partition_split([1,2,3,4],2,4) = [[1,2],[2,3],[3,4]]
    def partition_split(x,n,e):
        return np.array([np.array(x[i:i+n]) for i in range(0,e-n+1)])

    #remap y values from x index (reverse=False)
    #remap y values to x index (reverse=True)
    #example:
    # remap([1,2,0],["a","b","c"],reverse=False) = ["b","c","a"]
    # remap([1,2,0],["a","b","c"],reverse=True)  = ["c","a","b"]
    def remap(x,y,reverse=False):
        r = [None]*len(x)
        if reverse == False:
            for i in range(len(x)):
                r[i] = y[x[i]]
        else:
            for i in range(len(x)):
                r[x[i]] = y[i]
        return r

    #repeat array n times
    #example:
    # repeat([1,2],2) = [1,2,1,2]
    def repeat(x,n):
        r = [0]*(len(x)*n)
        for j in range(len(x)):
            for i in range(n):
                r[j*n+i] = x[j]
        return r

    #remove elements from x where y != 0
    #example:
    # rmm([1,2,3,4,5],[0,1,0,1,1]) = [1,3]
    def rmm(x,y):
        r = []
        for i in range(len(x)):
            if y[i] == 0:
                r.append(x[i])
        return r

    #sum of absolute differences
    def sad(x):
        def zsort(a,b):  #sort a,b using a
            z = sorted(zip(a,b), key=lambda pair: pair[0])
            a,b = zip(*z)
            return a,b
        n = len(x)
        idx = np.arange(n)
        x,idx = zsort(x,idx)   #sort x,idx by x (sort)
        s = sum(x)             #sum
        h = 0                  #running sum
        r = [0]*n
        for i in range(n):
            r[i] = x[i]*(2*i-n)+(s-2*h)
            h = h + x[i]
        idx, r = zsort(idx,r)  #sort idx,r by idx (unsort)
        return list(r)

    #shift by s and divide array by n point parts
    #example:
    # shift_divide([1,2,3,4],0) = [[], [], [], [], []]
    # shift_divide([1,2,3,4],1) = [[1], [2], [3], [4]]
    # shift_divide([1,2,3,4],2) = [[1, 2], [2, 3], [3, 4]]
    # shift_divide([1,2,3,4],3) = [[1, 2, 3], [2, 3, 4]]
    # shift_divide([1,2,3,4],4) = [[1, 2, 3, 4]]
    # shift_divide([1,2,3,4],5) = []
    def shift_divide(x,n,s=1):
        r = []
        for i in range(0,len(x)-n+1,s):
            temp = []
            for j in range(n):
                temp.append(x[i+j])
            r.append(temp)
        return r

    #shift by s and merge array
    #example:
    # shift_merge([[1,2,3],[1,2,3]]) = [1.0, 1.5, 2.5, 3.0]
    # shift_merge([[1,2,3],[1,2,3],[1,2,3]]) = [1.0, 1.5, 2.0, 2.5, 3.0]
    def shift_merge(x,s=1):
        r = []
        for i in range(0,len(x[0])+(len(x)-1)*s,1):
            #print("---sum---")
            sum = 0
            n = 0
            for j in range(0,i+1):
                if j < len(x) and i-j*s >= 0 and i-j*s < len(x[0]):
                    #print("adding layer " + str(j) + " , index " + str(i-j*s))
                    sum = sum + x[j][i-j*s]
                    n = n + 1
            r.append(sum/n)
        return r

    #slice leading zeros
    #example:
    # slice_zero([0,0,0]) = [0, 0, 0], []
    # slice_zero([1,0,0]) = [], [1, 0, 0]
    # slice_zero([0,1,0]) = [0], [1, 0]
    # slice_zero([0,0,1]) = [0, 0], [1]
    def slice_zero(x):
        i = 0
        while i < len(x):
            if x[i] != 0.0:
                break
            i = i + 1
        return x[:i],x[i:]

    #split array into positive and negative arrays
    #example:
    # split_zero([0,1,0,-2,2,0,-1,0]) = [1, 2], [-2, -1]
    def split_zero(x):
        p = []
        n = []
        for i in range(len(x)):
            if x[i] > 0:
                p.append(x[i])
            elif x[i] < 0:
                n.append(x[i])
        return p,n

    #split array evenly into c parts
    def split(x,c):
        n = len(x)
        quo = int(n/c)
        rem = n%c
        r = []
        for i in range(rem):
            r.append(x[i*(quo+1):(i+1)*(quo+1)])
        for i in range(rem,c):
            r.append(x[i*quo+rem:(i+1)*quo+rem])
        return r

    #sort x by y
    #example:
    # zipsort([7,8,9],[3,2,1]) = [9,8,7],[1,2,3]
    def zipsort(x,y):
        z = sorted(zip(y,x), key=lambda pair: pair[0])
        x = [x1 for _,x1 in z]
        y = [y1 for y1,_ in z]
        return x,y

    def test():
        y = []
        for i in range(10):
            y.append(np.random.uniform(0,1))
        ys = []
        for i in range(4):
            ys.append(y)
            y = Arr.interpolate(y,1,Arr.inter_qspline)
        xs = []
        for i in range(4):
            xs.append([j/(len(ys[i])-1) for j in range(0,len(ys[i]))])
        Mplt.plots(xs,ys)
        Mplt.show()
        Mplt.close()

class Cluster():
    def valid_metrics():
        return BallTree.valid_metrics

    #tsne
    #x: data
    #n_components: 2 for 2d, 3 for 3d
    #perplexity: perplexity (recommended: 5.0 - 50.0)
    #metric: distance metric ['hamming', 'dice', 'jaccard', 'matching', 'russellrao', 'euclidean', 'kulsinski', 'chebyshev', 'sokalmichener', 'rogerstanimoto', 'infinity', 'p', 'canberra', 'sokalsneath', 'l1', 'minkowski', 'l2', 'cityblock', 'braycurtis', 'manhattan']
    def tsne(x,n_components,perplexity=30.0,metric='euclidean'):
        if len(x[0]) >= n_components:
            return TSNE(n_components=n_components,perplexity=perplexity,metric=metric).fit_transform(x)
        else:
            return None

    #find optimal eps(distance) for dbscan
    #x: data (tsne)
    #n_neighbors: nth neighbor to examine
    #algorithm: nearest neighbor algorithm
    def distance(x,n_neighbors=5,algorithm='ball_tree'):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(x)
        distances, indices = nbrs.kneighbors(x)
        distances = sorted([d[-1] for d in distances])  #sort the nth nearest neighbor distance
        max_distance = distances[-1]
        #remove outliers
        out = Outlier.half(distances)
        i = 0
        while i < len(out):
            if out[i] != 0:
                break
            i = i + 1
        distances = distances[0:i]
        elbow_index,elbow_distance = Elbow.search(distances)
        return elbow_distance,max_distance

    #dbscan
    #x: data (tsne)
    #eps: distance
    #min_samples: minimum neighbors within eps to consider as cluster
    def dbscan(x,eps,minpts=5):
        dbscan = DBSCAN(eps=eps,min_samples=minpts)
        return dbscan.fit_predict(x)

    #dbscan with eps estimation (eps = elbow of k-distance graph with k=minpts-1)
    #minpts: (minpts >= 3) and (minpts >= dimensions + 1)
    def dbscan_eps(x,minpts):
        elbow_distance,max_distance = Cluster.distance(x,n_neighbors=minpts-1)
        return Cluster.dbscan(x,eps=elbow_distance,minpts=minpts)

    #local outlier factor
    #lof([[0,0],[0,1],[1,0],[1,1],[2,2]]) = [ 1  1  1  1 -1]
    #return 1 (inlier); -1 (outlier)
    def lof(x,n_neighbors=2,contamination=0.1):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination)
        return clf.fit_predict(x)

    #local outlier factor with partition division
    #x array to analyze
    #partition number to partition by
    #div number to divide partition by
    #n_neighbors number of nearest neighbors to use
    #return -1 for outliers and 1 for inliers
    def lof_part(x,partition=24,div=2,n_neighbors=2,contamination=0.1):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination)
        offset = partition/div
        lofs = []
        for i in range(div):
            lof_div = Arr.partition(x,n=partition,ci=int(offset*i),cf=int(offset*(i+1)))
            lof_div = clf.fit_predict(lof_div)
            lof_div = np.repeat(Arr.expand(lof_div,div,i,1),offset)
            lof_div = np.append(lof_div,[1]*(len(x)-len(lof_div)))
            lofs.append(lof_div)
        return lofs

class ConvexHull():
    plotflag = False
    fig = None
    ax = None

    #return the cross product of ao and ob
    #return positive if ao > ob > ba is counterclockwise
    #       negative if ao > ob > ba is clockwise
    #       zero if collinear
    def cross(a, o, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def prune_plot(points,i,a,b,c,d,x1,x2,y1,y2):
        if ConvexHull.plotflag == False:
            ConvexHull.plotflag = True
            ConvexHull.fig = plt.figure()
            ConvexHull.ax = ConvexHull.fig.add_subplot(111)
        fig = ConvexHull.fig
        ax = ConvexHull.ax
        ax.clear()
        plt.axis("equal")
        plt.scatter([p[0] for p in points[i:]],[p[1] for p in points[i:]],color="purple")
        plt.scatter([p[0] for p in points[:i]],[p[1] for p in points[:i]],color="pink")
        plt.scatter([points[i][0]],[points[i][1]],color="green")
        if not None in [a,b,c,d]:
            plt.plot([v[0] for v in [a,b,c,d,a]],[v[1] for v in [a,b,c,d,a]],color="yellow")
        for str,xy in list(zip(["a","c"],[a,c])):
            if xy != None:
                ax.annotate(str,xy=xy)
                ax.plot([xy[0]-0.7,xy[0]+0.7],[xy[1]-0.7,xy[1]+0.7],color="blue")
        for str,xy in list(zip(["b","d"],[b,d])):
            if xy != None:
                ax.annotate(str,xy=xy)
                ax.plot([xy[0]-0.7,xy[0]+0.7],[xy[1]+0.7,xy[1]-0.7],color="blue")
        plt.plot([x1,x1],[y1-1,y2+1],color="red")
        ax.annotate("x1",xy=[x1,y1-1])
        plt.plot([x2,x2],[y1-1,y2+1],color="red")
        ax.annotate("x2",xy=[x2,y1-1])
        plt.plot([x1-1,x2+1],[y1,y1],color="red")
        ax.annotate("y1",xy=[x1-1,y1])
        plt.plot([x1-1,x2+1],[y2,y2],color="red")
        ax.annotate("y2",xy=[x1-1,y2])
        plt.pause(.1)

    def gt(xm,da_xm,db_xm,dc_xm,dd_xm):
        for v in [da_xm,db_xm,dc_xm,dd_xm]:
            if v != None:
                if v > xm:
                    return False
        return True

    def lt(xm,da_xm,db_xm,dc_xm,dd_xm):
        for v in [da_xm,db_xm,dc_xm,dd_xm]:
            if v != None:
                if v < xm:
                    return False
        return True

    #prune points that cannot be part of the convex hull
    #c ___ b
    # |   |
    # |___|
    #d     a
    def prune(points):
        a = points[len(points)-1]  #max x min y max(x-y) bot right
        b = None                   #max x max y max(x+y) top right
        c = None                   #min x max y min(x-y) top left
        d = None                   #min x min y min(x+y) bot left
        da_xmy = points[len(points)-1][0]-points[len(points)-1][1]
        da_xpy = points[len(points)-1][0]+points[len(points)-1][1]
        db_xmy = None
        db_xpy = None
        dc_xmy = None
        dc_xpy = None
        dd_xmy = None
        dd_xpy = None
        x1 = a[0]  #left wall
        x2 = a[0]  #right wall
        y1 = a[1]  #bot wall
        y2 = a[1]  #top wall
        r1 = [points[len(points)-1]]
        i = len(points)-2
        while i >= 0:
            if not None in [a,b,c,d]:
                if points[i][0] > x1 and points[i][0] < x2:
                    if points[i][1] > y1 and points[i][1] < y2:
                        i = i - 1
                        continue
            xmy = points[i][0]-points[i][1]
            xpy = points[i][0]+points[i][1]
            if ConvexHull.gt(xpy,da_xpy,db_xpy,dc_xpy,dd_xpy):  #largest xpy (quadrant 1)
                b = points[i]
                db_xpy = xpy
                db_xmy = xmy
            elif ConvexHull.gt(xmy,da_xmy,db_xmy,dc_xmy,dd_xmy):
                a = points[i]
                da_xmy = xmy
                da_xpy = xpy
            elif ConvexHull.lt(xpy,da_xpy,db_xpy,dc_xpy,dd_xpy):  #smallest xpy (quadrant 3)
                d = points[i]
                dd_xpy = xpy
                dd_xmy = xmy
            elif ConvexHull.lt(xmy,da_xmy,db_xmy,dc_xmy,dd_xmy):
                c = points[i]
                dc_xmy = xmy
                dc_xpy = xpy
            if not None in [a,b,c,d]:
                x1 = max(c[0],d[0])  #left wall
                x2 = min(a[0],b[0])  #right wall
                y1 = max(a[1],d[1])  #bot wall
                y2 = min(b[1],c[1])  #top wall
            #ConvexHull.prune_plot(points,i,a,b,c,d,x1,x2,y1,y2)
            r1.append(points[i])
            i = i - 1
        r2 = []
        i = len(r1)-1
        while i >= 0:
            if r1[i][0] > x1 and r1[i][0] < x2:
                if r1[i][1] > y1 and r1[i][1] < y2:
                    i = i - 1
                    continue
            r2.append(r1[i])
            i = i - 1
        return r2

    #return the points of a convex hull
    def convex_hull(points):
        points = list(set(points))
        #print(len(points))
        if len(points) > 100:
            points = ConvexHull.prune(points)
        #print(len(points))
        points = sorted(points)
        if len(points) <= 1:
            return points
        lower = []
        for p in points:
            while len(lower) >= 2 and ConvexHull.cross(lower[-1], lower[-2], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and ConvexHull.cross(upper[-1], upper[-2], p) <= 0:
                upper.pop()
            upper.append(p)
        return list(reversed(lower[:-1] + upper[:-1]))

    #return the area of a polygon
    def polygon_area(points):
        area = 0
        j = len(points)-1
        for i in range(len(points)):
            area = area + (points[j][0]+points[i][0]) * (points[j][1]-points[i][1])
            j = i
        return area/2

    #return the area of a convex hull of a set of points
    def area(points):
        return ConvexHull.polygon_area(ConvexHull.convex_hull(points))

    def test():
        points = [(p[0],p[1]) for p in np.random.uniform(-1,1,size=(100,2)).tolist()]
        start_time = time.time()
        cpoints = ConvexHull.convex_hull(points)
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.figure()
        plt.axis('equal')
        plt.scatter([p[0] for p in points],[p[1] for p in points])
        plt.plot([p[0] for p in cpoints],[p[1] for p in cpoints],color="red")
        plt.show()

class Convolution():
    #product of a*b in the time domain
    def _tproduct(a,b):
        al = len(a)
        bl = len(b)
        r = [0]*(al-bl+1)
        for i in range(al-bl+1):
            r[i] = sum([a[i:i+bl][j]*b[j] for j in range(bl)])
        return r

    #product of a*b in the frequency domain
    def _fproduct(a,b):
        p = a*b
        ifft = np.fft.ifft(p)
        r = np.real(np.fft.fftshift(ifft)).tolist()
        for i in range(len(r)):
            if Proximate.zero(r[i]):
                r[i] = 0.0
        r = r[len(r)-1:len(r)] + r[0:len(r)-1]
        return p,ifft,r

    #apply convolution
    #x: data set
    #k: kernel
    #endpoints: "expand":   expand endpoints   (len(r) = len(x)+(len(k)-1))
    #           "keep":     keep endpoints     (len(r) = len(x))
    #           "contract": contract endpoints (len(r) = len(x)-(len(k)-1))
    #normalize: normalize by dividing by conv(uniform,kernel)
    #debug: plot
    #usefft: use fft - fast O(n*log(n)) vs use loops - slow O(n**2)
    def apply(x,k,endpoints="keep",normalize=True,debug=False,usefft=True):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(k, np.ndarray):
            k = k.tolist()
        if not np.isfinite(x).all():
            str = "error: data set contains invalid values"
            print(str)
            raise Exception(str)
        if not np.isfinite(k).all():
            str = "error: kernel contains invalid values"
            print(str)
            raise Exception(str)
        xl = len(x)
        kl = len(k)
        if xl < kl:
            str = "error: data set must be longer than kernel length"
            print(str)
            raise Exception(str)
        if kl%2 == 0:
            str = "error: kernel length must be odd"
            print(str)
            raise Exception(str)
        if not usefft:
            x_pad = kl-1
            x = [0]*x_pad + x + [0]*x_pad
            k_pad = int(kl/2)
            xk_r = Convolution._tproduct(x,k)
            if normalize:
                u = [0]*x_pad + [1]*xl + [0]*x_pad
                uk_r = Convolution._tproduct(u,k)
                xk_r = [xk_r[i] if uk_r[i] == 0 else xk_r[i]/uk_r[i] for i in range(len(xk_r))]
            if debug:
                ts = ["x","k","xk_r"]
                ys = [x,k,xk_r]
                fig = plt.figure()
                ax = [fig.add_subplot(310+i) for i in range(1,4)]
                [[ax[i].set_ylabel(ts[i]),ax[i].plot(np.arange(len(y)),y)] for (i,y) in enumerate(ys)]
                plt.show()
                plt.close()
            if endpoints == "expand":
                return xk_r
            elif endpoints == "keep":
                return xk_r[k_pad:len(xk_r)-k_pad]
            elif endpoints == "contract":
                return xk_r[k_pad*2:len(xk_r)-k_pad*2]
        else:
            x_rem = 1-xl%2
            x_pad = int(kl/2)  #endpoints
            x = [0]*x_rem + [0]*x_pad + x + [0]*x_pad   #make len(x) odd
            k_pad = int((len(x)-kl)/2)
            k = [0]*k_pad + k + [0]*k_pad               #make len(k) odd
            x_f = np.fft.fft(x)
            k_f = np.fft.fft(k)
            xk_f, xk_ifft, xk_r = Convolution._fproduct(x_f,k_f)
            if normalize:
                u = [0]*x_rem + [0]*x_pad + [1]*xl + [0]*x_pad  #uniform mask
                u_f = np.fft.fft(u)
                uk_f, uk_ifft, uk_r = Convolution._fproduct(u_f,k_f)
                xk_r = [xk_r[i] if uk_r[i] == 0 else xk_r[i]/uk_r[i] for i in range(len(xk_r))]
            if debug:
                ts = ["x","k","x_f","k_f","xk_f","xk_ifft","xk_r"]
                ys = [x,k,x_f,k_f,xk_f,xk_ifft,xk_r]
                fig = plt.figure()
                ax = [fig.add_subplot(710+i) for i in range(1,8)]
                [[ax[i].set_ylabel(ts[i]),ax[i].plot(np.arange(len(y)),np.real(y)),ax[i].plot(np.arange(len(y)),np.imag(y))] for (i,y) in enumerate(ys)]
                plt.show()
                plt.close()
            if endpoints == "expand":
                return xk_r[x_rem:len(xk_r)]
            elif endpoints == "keep":
                return xk_r[x_rem+x_pad:len(xk_r)-x_pad]
            elif endpoints == "contract":
                return xk_r[x_rem+x_pad*2:len(xk_r)-x_pad*2]

    #perform convolution n times (r[i] = ith convolution)
    def apply_n(x,k,n,endpoints="keep",normalize=True):
        r = [0]*(n+1)
        r[0] = x
        for i in range(n):
            r[i+1] = Convolution.apply(r[i],k,endpoints=endpoints,normalize=normalize)
        return r

    def test():
        y = np.sin(np.array([i/25 for i in range(0,500)])) + 0.5*(np.random.rand(500)*2-1)
        for i in np.random.randint(0,500,100):
            y[i] = y[i] + np.random.rand()*2-1
        plt.figure()
        plt.plot(np.arange(len(y)),y,label="original")
        c = Convolution.apply(y,Kernel.uniform(21),endpoints="keep")
        plt.plot(np.arange(len(c)),c,label="uniform")
        c = Convolution.apply_n(y,Kernel.kaiser(21),200,endpoints="keep")
        for i in [1] + list(range(10,201,10)):
            s = int((len(y)-len(c[i]))/2)
            plt.plot(np.arange(s,s+len(c[i])),c[i],label="c"+str(i))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

class CUSUM():
    #x: data
    #u: estimated mean
    #h: threshold
    #k: rise per unit
    def table(x,u,h,k):
        cusum = [0] * (len(x)+1)
        s_hi = [0] * (len(x)+1)
        s_lo = [0] * (len(x)+1)
        signal = [0] * (len(x)+1)
        for i in range(len(x)):
            d = x[i] - u
            cusum[i+1] = cusum[i] + d
            s_hi[i+1] = max(0,s_hi[i] + d - k)
            s_lo[i+1] = max(0,s_lo[i] - d - k)
            if s_hi[i+1] > h or s_lo[i+1] > h:
                signal[i+1] = 1
        return cusum,s_hi,s_lo,signal

    def test():
        x = 324.925, 324.675, 324.725, 324.350, 325.350, 325.225, 324.125, 324.525, 325.225, 324.600, 324.625, 325.150, 328.325, 327.250, 327.825, 328.500, 326.675, 327.775, 326.875, 328.350
        cusum,s_hi,s_lo,signal = CUSUM.table(x,325,4.1959,0.03175)
        print(cusum)
        print(s_hi)
        print(s_lo)
        print(signal)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(len(x)),x)
        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(len(cusum)),cusum)
        ax2.plot(np.arange(len(s_hi)),s_hi,color="green")
        ax2.plot(np.arange(len(s_lo)),s_lo,color="yellow")
        ax2.plot(np.arange(len(signal)),signal,color="red")
        plt.show()
        plt.close()

class Dispersion():
    #standard deviation
    #https://en.wikipedia.org/wiki/Standard_deviation
    def std(y):
        n = len(y)
        u = np.mean(y)
        s = 0
        for i in range(n):
            s = s + (y[i]-u)*(y[i]-u)
        return math.sqrt(s/n)

    #pooled standard deviation
    #https://en.wikipedia.org/wiki/Pooled_variance
    #n: size
    #u: mean
    #s: standard deviation
    def std_pooled(n1,u1,s1,n2,u2,s2):
        n = n1+n2
        u = (n1*u1+n2*u2)/n
        d1 = u1-u
        d2 = u2-u
        sub1 = n1*(s1**2+d1**2)
        sub2 = n2*(s2**2+d2**2)
        s = math.sqrt((sub1+sub2)/n)
        return s

    def percentile(y, p):
        ys = sorted(y)
        return ys[int(round(p * len(ys) + 0.5))-1]

    #interquartile range
    def iqrange(y,k=1.5):
        ys = sorted(y)
        q1 = Dispersion.percentile(ys,0.25)
        q3 = Dispersion.percentile(ys,0.75)
        iqr = q3-q1
        return q1-k*iqr,q3+k*iqr

    #mean absolute difference
    #dim: dimensionality (use dim=1 for ordered data such as time series)
    #https://en.wikipedia.org/wiki/Mean_absolute_difference
    def mean_abs_diff(y,dim=1):
        n = len(y)
        s = 0
        if dim == 1:
            for i in range(0,n-1):
                s = s + abs(y[i]-y[i+1])
            return s/(n-1)
        elif dim == 2:
            for i in range(0,n):
                for j in range(i+1,n):
                    s = s + abs(y[i]-y[j])
            return (s*2)/(n*n)

    #mean absolute deviation
    #https://en.wikipedia.org/wiki/Average_absolute_deviation
    def mean_abs_dev(y,center="mean"):
        y = list(y)
        n = len(y)
        if center == "mean":
            u = np.mean(y)
        elif center =="median":
            u = np.median(y)
        for i in range(n):
            y[i] = abs(y[i]-u)
        return np.mean(y)

    #median absolute deviation
    #https://en.wikipedia.org/wiki/Median_absolute_deviation
    def median_abs_dev(y,center="median"):
        y = list(y)
        n = len(y)
        if center == "median":
            u = np.median(y)
        elif center == "mean":
            u = np.mean(y)
        for i in range(n):
            y[i] = abs(y[i]-u)
        return np.median(y)

    def test():
        y = [1,2,3,4,5]
        print(np.std(y))
        print(Dispersion.std(y))
        print(Dispersion.iqrange(y))
        print(Dispersion.mean_abs_diff(y,dim=1))
        print(Dispersion.mean_abs_diff(y,dim=2))
        print(Dispersion.mean_abs_dev(y))
        print(Dispersion.median_abs_dev(y))
        y1 = np.random.uniform(0,100,100).tolist()
        n1 = len(y1)
        u1 = np.mean(y1)
        s1 = np.std(y1)
        y2 = np.random.uniform(0,100,100).tolist()
        n2 = len(y2)
        u2 = np.mean(y2)
        s2 = np.std(y2)
        print(Dispersion.std(y1+y2))
        print(Dispersion.std_pooled(n1,u1,s1,n2,u2,s2))

class Elbow():
    #if n > 1
    #    compute the elbow index using the starting index
    #    use the elbow index as the new starting index
    #    repeat
    #if maxslope == True
    #    calculate the slope using starting point and iterating through data points
    #    use data point with the max slope as the new end point
    #    the point with the max slope is the first point that is reachable from the starting point by a clockwise direction
    #if weight > 1
    #    weight farther x values higher
    #    calculate distance as (1+(weight-1)*(x/xmax))*distance
    #if log == True
    #    calculate log elbow
    def search(z,n=1,maxslope=False,weight=1,log=False,debug=False):
        x1 = [[] for _ in range(n+1)]
        y1 = [[] for _ in range(n+1)]
        d1 = [[] for _ in range(n+1)]
        if maxslope:
            start = int(len(z)/2)  #starting index (first half of data cannot be an end point)
            s = [0]*len(z)
            for i in range(start,len(z)):
                s[i] = (z[i]-z[0])/i
            s_max = max(s)
            i_max = 0
            for i in range(len(z)-1,-1,-1):
                if s[i] == s_max:
                    i_max = i
                    break
            z = z[:i_max+1]
        if log:
            k = 0
            while k < len(z):  #skip undefined log(z[k])
                if z[k] > 0:
                    break
                i = i + 1
            y1[0] = [math.log(v) for v in z[k:]]
            x1[0] = np.arange(len(y1[0]))
            d1[0] = 0
        else:
            y1[0] = z
            x1[0] = np.arange(len(y1[0]))
            d1[0] = 0
        idx = [0]*(n+1)
        for i in range(0,n):
            y1[i+1] = y1[i][idx[i]:]
            x1[i+1] = np.arange(len(y1[i+1]))
            d = Vec.rdist(x1[i+1],y1[i+1])
            if weight == 1:
                d1[i+1] = d
            else:
                d1[i+1] = [(1+(weight-1)*(i/(len(d)-1)))*d[i] for i in range(len(d))]
            idx[i+1] = d1[i+1].index(max(d1[i+1]))  #max rejection distance
        if log:
            elbow_index = sum(idx) + k
            elbow_value = z[elbow_index]
        else:
            elbow_index = sum(idx)
            elbow_value = z[elbow_index]
        y1 = y1[1:]
        x1 = x1[1:]
        d1 = d1[1:]
        if debug:
            print("elbow index:",elbow_index)
            print("elbow value:",elbow_value)
            fig, axes = plt.subplots(n,2)
            for i in range(n):
                axes[i, 0].set_title("elbow")
                axes[i, 0].plot(x1[i], y1[i])
                axes[i, 0].plot([x1[i][0],x1[i][-1]], [y1[i][0],y1[i][-1]])
                axes[i, 0].plot([x1[i][idx[i+1]],x1[i][-1]], [y1[i][idx[i+1]],y1[i][-1]])
                axes[i, 1].set_title("elbow distance")
                axes[i, 1].plot(x1[i], d1[i])
                hull = ConvexHull.convex_hull([(x1[i][j],y1[i][j]) for j in range(len(x1[i]))])
                axes[i, 0].scatter([hull[i][0] for i in range(len(hull))], [hull[i][1] for i in range(len(hull))], color="red")
            plt.show()
            plt.close()
        return elbow_index,elbow_value

    def test():
        y = [v/100 + 5*(1+math.sin(v/100)) for v in range(5000)] + [55+math.log(v+2,2) for v in range(1000)]
        Elbow.search(y,n=3,debug=True)

class Entropy():
    def _phi(u, m, r):
        n = len(u)
        n_sub = n - m + 1  #number of subsequences
        z = [[u[j] for j in range(i, i + m)] for i in range(n_sub)]
        clist = [0]*n_sub
        for i in range(n_sub):
            c = 0
            for j in range(n_sub):
                c = c + int(Metrics.chebyshev(z[i],z[j]) <= r)
            clist[i] = c/n_sub
        return sum(np.log(clist))/n_sub

    #approximate entropy (https://en.wikipedia.org/wiki/Approximate_entropy)
    #u: data
    #m: subsequence length
    #r: filter
    def apen(u, m, r):
        return abs(Entropy._phi(u,m+1,r) - Entropy._phi(u,m,r))

    def entropy(x):
        d = dict()
        for i in range(len(x)):
            d[x[i]] = 1
        return math.log2(len(d))

    def test():
        print(Entropy.entropy([i for i in range(2**8)]))
        print(Entropy.apen([10,20,10,20,10,20,10,20,10,20,10,20],2,3))
        print(Entropy.apen([10,10,20,10,20,20,20,10,10,20,10,20],2,3))

class Fc():
    #n choose r
    def ncr(n, r):
        r = min(r, n-r)
        if r <= 0:
            return 1
        numer = functools.reduce(operator.mul, range(n, n-r, -1))
        denom = functools.reduce(operator.mul, range(1, r+1))
        return numer//denom

    #generalized harmonic number
    def ghn(n, m, q=0):
        s = 0
        for i in range(1,n+1):
            s = s + (i+q)**(-m)
        return s

    #bessel function (a:order, M:approximation of inf)
    def bf(x,a=0,M=100):
        return sum([((-1)**m)/(math.factorial(m)*math.gamma(m+a+1))*((x/2)**(2*m+a)) for m in range(M)])

    #modified bessel function (a:order, M:approximation of inf)
    def mbf(x,a=0,M=100):
        return sum([1/(math.factorial(m)*math.gamma(m+a+1))*((x/2)**(2*m+a)) for m in range(M)])

    #hyperbolic cosine
    def cosh(x):
        return (math.exp(x)+math.exp(-x))/2

    #inverse hyperbolic cosine
    def acosh(x):
        return math.log(x+math.sqrt(x**2-1))

    #mth order chebyshev polynomial
    def cp(m,x,cosh=True):
        if cosh:
            if abs(x) <= 1:
                return math.cos(m*math.acos(x))
            elif x > 1:
                return math.cosh(m*math.acosh(x))
            elif x < -1:
                return ((-1)**m)*math.cosh(m*math.acosh(-x))
        else:
            if abs(x) <= 1:
                return math.cos(m*math.acos(x))
            else:
                t = math.sqrt(x**2-1)
                return 0.5*(((x-t)**m)+((x+t)**m))

    #chebyshev polynomial coefficient matrix
    def cpcm(n):
        c = [[0 for _ in range(n)] for _ in range(n)]
        c[1][1] = 1
        c[2][2] = 1
        #c[i][1] = -c[i-2][1]               #i=3,4...
        #c[i][j] = 2*c[i-1][j-1]            #i=j=3,4...
        #c[i][j] = 2*c[i-1][j-1]-c[i-2][j]  #i>j=2,3...
        for i in range(3,n):
            c[i][1] = -c[i-2][1]
        for i in range(3,n):
            c[i][i] = 2*c[i-1][i-1]
        for j in range(2,n,1):
            for i in range(j+2,n,2):
                c[i][j] = 2*c[i-1][j-1]-c[i-2][j]
        return c

    #falling factorial
    #ff(x,0) = 1
    #ff(x,1) = x
    #ff(x,2) = x*(x-1)
    def ff(x,n,gamma=True):
        if gamma:
            return math.gamma(x+1)/math.gamma(x-n+1)
        else:
            p = 1  #product
            for i in range(0,n):
                p=p*(x-i)
            return p

    #rising factorial
    #fr(x,0) = 1
    #fr(x,1) = x
    #fr(x,2) = x*(x+1)
    def fr(x,n,gamma=True):
        if gamma:
            return math.gamma(x+n)/math.gamma(x)
        else:
            p = 1  #product
            for i in range(0,n):
                p=p*(x+i)
            return p

    #hypergeometric function
    def hgf(a,b,c,z,N=100):
        s = 1   #sum
        an = 1  #rising factorial
        bn = 1  #rising factorial
        cn = 1  #rising factorial
        fn = 1  #factorial
        for n in range(0,N):
            an = an*(a+n)
            bn = bn*(b+n)
            cn = cn*(c+n)
            fn = fn*(n+1)
            s = s + (an*bn/cn)*((z**(n+1))/fn)
        return s

    #gegenbauer polynomial
    def gp(n,a,z):
        f = (math.gamma(2*a+n)/math.gamma(2*a))/math.factorial(n)
        return f*Fc.hgf(-n,2*a+n,a+0.5,0.5*(1-z))

    #jacobi polynomial
    def jp(n,a,b,z):
        f = (math.gamma(a+1+n)/math.gamma(a+1))/math.factorial(n)
        return f*Fc.hgf(-n,1+a+b+n,a+1,0.5*(1-z))

    #integral approximation
    def integral(f,x1,x2,dx):
        s = 0
        i = x1
        while i+dx <= x2-dx:       #add inner values
            i = i + dx
            s = s + f(i)
        if i != x1:
            i = i + dx
            s = s + (f(x1)+f(i))/2
        r = (f(i)+f(x2))*(x2-i)/2  #remainder
        return s*dx+r

    #beta function
    def beta(x,y):
        return math.gamma(x)*math.gamma(y)/math.gamma(x+y)

    #gamma function approximation
    #gamma(z) = (z-1)!
    def gamma(z,approx="integral"):
        if z == 0:
            print("domain error:",z)
            return None
        if z < 0 and float(z).is_integer():
            print("domain error:",z)
            return None
        if z > 0:
            z0 = z
        elif z < 0:
            z0 = 1-z
        if approx == "integral":
            g = Fc.integral(lambda x: 0 if x == 0 else (x**(z0-1))*math.exp(-x),0,100,0.001)
        if z > 0:
            return g
        elif z < 0:
            return (1/g)*(math.pi/math.sin(math.pi*z))

    def test():
        #cpcm test
        print(np.array(Fc.cpcm(10)))
        #integral test
        print("x**2 from 0 to 1 =",Fc.integral(lambda x: x*x,0,1,0.01))
        print("x**3 from 0 to 1 =",Fc.integral(lambda x: x*x*x,0,1,0.01))
        #gamma test
        x = []
        g1 = []
        g2 = []
        for i in range(-30,30):
            v = i/10
            x.append(v)
            if v <= 0 and float(v).is_integer():
                g1.append(None)
                g2.append(None)
            else:
                g1.append(math.gamma(v))
                g2.append(Fc.gamma(v))
        plt.figure()
        ax = plt.subplot()
        ax.plot(x,g1,color="green")
        ax.scatter(x,g2,color="red")
        plt.show()
        plt.close()
        #hgf test
        n = 100
        x = np.arange(n)
        #y1 = [1/(1-i/n) for i in range(0,n)]
        #y2 = [Fc.hgf(1,1,1,i/n) for i in range(0,n)]
        y1 = [1/((1-i/n)**2) for i in range(0,n)]
        y2 = [Fc.hgf(1,2,1,i/n) for i in range(0,n)]
        plt.figure()
        ax = plt.subplot()
        ax.plot(x,y1,color="green")
        ax.scatter(x,y2,color="red")
        plt.show()
        plt.close()

#https://en.wikipedia.org/wiki/Binary_classification
#https://en.wikipedia.org/wiki/Precision_and_recall
#https://en.wikipedia.org/wiki/Accuracy_paradox
#true  = 1
#false = 0
class Fscore():
    def _count(test_value,actual_value,test,actual):
        count = 0
        for i in range(len(test)):
            if test[i] == test_value and actual[i] == actual_value:
                count = count + 1
        return count

    #true positives
    def tp(test,actual):
        return Fscore._count(1,1,test,actual)

    #false positives - type 1 error
    def fp(test,actual):
        return Fscore._count(1,0,test,actual)

    #true negatives
    def tn(test,actual):
        return Fscore._count(0,0,test,actual)

    #false negatives - type 2 error
    def fn(test,actual):
        return Fscore._count(0,1,test,actual)

    #true positive rate - hit rate - recall - how complete
    def TPR(test,actual):
         tp = Fscore.tp(test,actual)
         fn = Fscore.fn(test,actual)
         if tp == 0:
             return 0
         else:
             return tp/(tp+fn)

    def recall(test,actual):
        return Fscore.TPR(test,actual)

    #true negative rate - specificity
    def TNR(test,actual):
         tn = Fscore.tn(test,actual)
         fp = Fscore.fp(test,actual)
         if tn == 0:
             return 0
         else:
             return tn/(tn+fp)

    def specificity(test,actual):
        return Fscore.TNR(test,actual)

    #positive predictive value - precision - how useful
    def PPV(test,actual):
         tp = Fscore.tp(test,actual)
         fp = Fscore.fp(test,actual)
         if tp == 0:
             return 0
         else:
             return tp/(tp+fp)

    def precision(test,actual):
        return Fscore.PPV(test,actual)

    #negative predictive value
    def NPV(test,actual):
         tn = Fscore.tn(test,actual)
         fn = Fscore.fn(test,actual)
         if tn == 0:
             return 0
         else:
             return tn/(tn+fn)

    #false negative rate - miss rate
    def FNR(test,actual):
        return 1-Fscore.TPR(test,actual)

    #false positive rate - fall out
    def FPR(test,actual):
        return 1-Fscore.TNR(test,actual)

    #false discovery rate
    def FDR(test,actual):
        return 1-Fscore.PPV(test,actual)

    #false omission rate
    def FOR(test,actual):
        return 1-Fscore.NPV(test,actual)

    #accuracy
    def ACC(test,actual):
        tp = Fscore.tp(test,actual)
        tn = Fscore.tn(test,actual)
        fp = Fscore.fp(test,actual)
        fn = Fscore.fn(test,actual)
        return (tp+tn)/(tp+tn+fp+fn)

    def accuracy(test,actual):
        return Fscore.ACC(test,actual)

    #f measure - harmonic mean of precision and recall
    def f(test,actual):
        ppv = Fscore.PPV(test,actual)
        tpr = Fscore.TPR(test,actual)
        if ppv == 0 and tpr == 0:
            return 0.0
        else:
            return 2*ppv*tpr/(ppv+tpr)

    #g measure - geometric mean of precision and recall
    def g(test,actual):
        ppv = Fscore.PPV(test,actual)
        tpr = Fscore.TPR(test,actual)
        return math.sqrt(ppv*tpr)

    def print(test,actual):
        print("accuracy: ",Fscore.ACC(test,actual))
        print("precision:",Fscore.PPV(test,actual))
        print("recall:   ",Fscore.TPR(test,actual))
        print("f measure:",Fscore.f(test,actual))
        print("g measure:",Fscore.g(test,actual))

    def test():
        print("-----")
        print("9700 true negatives")
        print("150 true positives")
        print("150 false positives")
        print("50 false negatives")
        test = [0]*9700+[1]*150+[1]*150+[0]*50
        actual = [0]*9700+[1]*150+[0]*150+[1]*50
        Fscore.print(test,actual)
        print("-----")
        print("9850 true negatives")
        print("150 false negatives")
        test = [0]*9850+[0]*150
        actual = [0]*9850+[1]*150
        Fscore.print(test,actual)

class KDE():
    #calculate score at v
    #v: sample
    #x: data set
    #h: bandwidth
    #k: kernel function (pdf)
    def score(v,x,h,k=None):
        if k == None:
            k = KDE.gaussian
        n = len(x)  #sample size
        s = 0
        for i in range(n):
            s = s + k((v-x[i])/h)
        return s/(n*h)

    #n: sample size
    #xmin: min sample
    #xmax: max sample
    #x: data set
    #h: bandwidth
    #k: kernel function (pdf)
    def kde(n,xmin,xmax,x,h,k=None):
        w = (xmax-xmin)/n
        xr = [xmin+i*w for i in range(n+1)]
        yr = [KDE.score(xr[i],x,h,k) for i in range(n+1)]
        return w,xr,yr

    #n: sample size
    #xmin: min sample
    #xmax: max sample
    #x: data set
    #h: bandwidth
    #k: kernel function (pdf)
    def kde2(n,xmin,xmax,x,h,k="gaussian"):
        w = (xmax-xmin)/n
        xr = [xmin+i*w for i in range(n+1)]
        kd = KernelDensity(kernel=k, bandwidth=h).fit([[v] for v in x])
        log_dens = kd.score_samples([[v] for v in xr])
        yr = np.exp(log_dens)
        return w,xr,yr

    def uniform(x):
        if abs(x) > 1:
            return 0
        return 0.5

    def triangular(x):
        if abs(x) > 1:
            return 0
        return 1-abs(x)

    def epanechnikov(x):
        if abs(x) > 1:
            return 0
        t = 1-x*x
        return 3/4*t

    def biweight(x):
        if abs(x) > 1:
            return 0
        t = 1-x*x
        return 15/16*t*t

    def triweight(x):
        if abs(x) > 1:
            return 0
        t = 1-x*x
        return 35/32*t*t*t

    def tricube(x):
        if abs(x) > 1:
            return 0
        t = 1-abs(x)*abs(x)*abs(x)
        return 70/81*t*t*t

    def gaussian(x):
        return math.exp(-x*x/2)/math.sqrt(2*math.pi)

    def cosine(x):
        if abs(x) > 1:
            return 0
        return math.pi/4*math.cos(math.pi/2*x)

    def logistic(x):
        return 1/(math.exp(x)+2+math.exp(-x))

    def sigmoid(x):
        return 2/(math.pi*(math.exp(x)+math.exp(-x)))

    def silverman(x):
        t = abs(x)/math.sqrt(2)
        return math.exp(-t)/2*math.sin(t+math.pi/4)

    def test():
        x = [0,1,2,3,7,7,7,7.3,7.6,7.9,10]
        w1,x1,y1 = KDE.kde(1000,-1,11,x,0.1,KDE.gaussian)
        w2,x2,y2 = KDE.kde2(1000,-1,11,x,0.1,"gaussian")
        print("1 =",Arr.area(y1,w1))
        print("1 =",Arr.area(y2,w2))
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(x1,y1)
        ax2 = fig.add_subplot(212,sharex=ax1)
        ax2.plot(x2,y2)
        plt.show()
        plt.close()

class Kernel():

    def kernels():
        d = OrderedDict()
        #f kernels
        d["f010"]            = Kernel.f010
        d["f011"]            = Kernel.f011
        d["f110"]            = Kernel.f110
        d["f111"]            = Kernel.f111
        #kernels without parameters
        d["lrdiff"]          = Kernel.lrdiff
        d["lshift"]          = Kernel.lshift
        d["rshift"]          = Kernel.rshift
        d["rect"]            = Kernel.rect
        d["tri"]             = Kernel.tri
        d["parzen"]          = Kernel.parzen
        d["welch"]           = Kernel.welch
        d["bohman"]          = Kernel.bohman
        #kernels with parameters
        d["sin"]             = Kernel.sin
        d["cos"]             = Kernel.cos
        d["gaussian"]        = Kernel.gaussian
        d["cgaussian"]       = Kernel.cgaussian
        d["tukey"]           = Kernel.tukey
        d["plancktaper"]     = Kernel.plancktaper
        d["kaiser"]          = Kernel.kaiser
        d["dolphchebyshev"]  = Kernel.dolphchebyshev
        d["dirichlet"]       = Kernel.dirichlet
        d["fejer"]           = Kernel.fejer
        d["ricker"]          = Kernel.ricker
        d["exponential"]     = Kernel.exponential
        d["lanczos"]         = Kernel.lanczos
        #finite PMF without parameters
        d["uniform"]         = Kernel.uniform
        d["triangular"]      = Kernel.triangular
        d["rademacher"]      = Kernel.rademacher
        d["benford"]         = Kernel.benford
        #finite PMF with parameters
        d["zipf"]            = Kernel.zipf
        d["skellam"]         = Kernel.skellam
        d["betabinomial"]    = Kernel.betabinomial
        d["binomial"]        = Kernel.binomial
        d["hypergeometric"]  = Kernel.hypergeometric
        #infinite PMF
        d["nbinomial"]       = Kernel.nbinomial
        d["nhypergeometric"] = Kernel.nhypergeometric
        d["geometric"]       = Kernel.geometric
        d["poisson"]         = Kernel.poisson
        d["borel"]           = Kernel.borel
        return d

    #PMF mean (center of mass)
    def mean(k):
        return sum([i*k[i] for i in range(len(k))])

    #PMF variance
    def var(k):
        return sum([i*i*k[i] for i in range(len(k))]) - (Kernel.mean(k)**2)

    #find [0,1,0]
    def f010(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [0]*(k-1) + [-1,1,-1] + [0]*(k-1)

    #find [0,1,1]
    def f011(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [0]*(k-1) + [1,1,-1] + [0]*(k-1)

    #find [1,1,0]
    def f110(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [0]*(k-1) + [-1,1,1] + [0]*(k-1)

    #find [1,1,1]
    def f111(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [0]*(k-1) + [1,1,1] + [0]*(k-1)

    #slope detection kernel
    def lrdiff(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [1]*k + [0] + [-1]*k

    #left shift kernel
    def lshift(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [1]*k + [1] + [0]*k

    #right shift kernel
    def rshift(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [0]*k + [1] + [1]*k

    #rectangular window (sum kernel)
    #b-spline with k = 1
    def rect(N):
        N = N + (1-N%2)
        return [1 for i in range(N)]

    #triangular window
    #b-spline with k = 2 (convolution of rect with rect)
    #type: ("-": start at 0, "+" do not start at 0)
    #example:
    # tri(3,type="-") = [0.0, 1.0, 0.0]
    # tri(3,type="+") = [0.5, 1.0, 0.5]
    def tri(N,type="-"):
        N = N + (1-N%2)
        k1 = int((N-1)/2)
        if type == "-":
            k2 = int((N-1)/2)
        elif type == "+":
            k2 = int((N+1)/2)
        return [1-abs((i-k1)/k2) for i in range(N)]

    #parzen window
    #b-spline with k = 4 (convolution of tri with tri)
    def parzen(N):
        N = N + (1-N%2)
        r = []
        for i in range(-N,N):
            if 0 <= abs(i) and abs(i) <= N/4:
                r.append(1-6*((i/(N/2))**2)*(1-abs(i)/(N/2)))
            elif N/4 < abs(i) and abs(i) <= N/2:
                r.append(2*((1-abs(i)/(N/2))**3))
        return r

    #welch window (parabolic kernel)
    #type: ("-": start at 0, "+" do not start at 0)
    def welch(N,type="-"):
        N = N + (1-N%2)
        k1 = int((N-1)/2)
        if type == "-":
            k2 = (N-1)/2
        elif type == "+":
            k2 = (N+1)/2
        return [1-(((i-k1)/k2)**2) for i in range(N)]

    #bohman window (autocorrelation of sine window)
    def bohman(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [(1-abs(i/k))*math.cos(math.pi*abs(i/k))+(1/math.pi)*math.sin(math.pi*abs(i/k)) for i in range(-k,k+1)]

    #sine window
    #a: power (0:rect, 1:sin, 2:hann)
    def sin(N,a=1):
        N = N + (1-N%2)
        return [math.sin(math.pi*i/(N-1))**a for i in range(N)]

    #cosine sum window
    #a: coefficients ([1]: rect, [1/2,1/2]: hann)
    #type: window type
    def cos(N,a=[1/2,1/2],type=None):
        N = N + (1-N%2)
        if type == "hann":
            a = [1/2,1/2]
        elif type == "hamming":
            a = [25/46,21/46]
        elif type == "blackman":
            #a = [(1-c)/2,1/2,c/2]
            a = [7938/18608,9240/18608,1430/18608]
        elif type == "nuttall":
            a = [0.355768,0.487396,0.144232,0.012604]
        elif type == "blackman-nuttall":
            a = [0.3635819,0.4891775,0.1365995,0.0106411]
        elif type == "blackman-harris":
            a = [0.35875,0.48829,0.14128,0.01168]
        elif type == "flattop":
            a = [1,1.93,1.29,0.388,0.028]
        return [sum([((-1)**k)*a[k]*math.cos(2*math.pi*k*i/(N-1)) for k in range(len(a))]) for i in range(N)]

    #gaussian kernel
    #s: standard deviation
    #p: power
    def gaussian(N,s=0.4,p=2):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [math.exp(-0.5*(((i-k)/(s*k))**p)) for i in range(N)]

    #approximate confined gaussian window
    #s: temporal width
    def cgaussian(N,s=0.4):
        N = N + (1-N%2)
        def g(x):  #gaussian function
            exp = -1*(((x-(N-1)/2)/(2*s))**2)
            return math.exp(exp)
        f = g(-0.5)/(g(-0.5+N)+g(-0.5-N))
        return [g(i)-f*(g(i+N)+g(i-N)) for i in range(N)]

    #tukey kernel
    #a: parameter (0: rect, 1: hann)
    def tukey(N,a=0.5):
        N = N + (1-N%2)
        k = N-1
        r = []
        for i in range(N):
            if 0 <= i and i < a*k/2:
                t = math.pi*((2*i)/(a*k)-1)
                r.append(0.5*(1+math.cos(t)))
            elif a*k/2 <= i and i <= k*(1-a/2):
                r.append(1)
            elif k*(1-a/2) < i and i <= k:
                t = math.pi*((2*i)/(a*k)-2/a+1)
                r.append(0.5*(1+math.cos(t)))
        return r

    #planck taper window
    #e: parameter
    def plancktaper(N,e=0.1):
        N = N + (1-N%2)
        k = N-1
        r = []
        for i in range(N):
            if 0 < i and i < e*k:
                d = (2*i)/k-1
                z = 2*e*(1/(1+d)+1/(1-2*e+d))
                r.append(1/(math.exp(z)+1))
            elif e*k <= i and i <= (1-e)*k:
                r.append(1)
            elif (1-e)*k < i and i < k:
                d = (2*i)/k-1
                d = -1*d
                z = 2*e*(1/(1+d)+1/(1-2*e+d))
                r.append(1/(math.exp(z)+1))
            else:
                r.append(0)
        return r

    #kaiser window
    #approximate dpss (discrete prolate spheroidal sequence) window
    #a: parameter
    def kaiser(N,a=3):
        N = N + (1-N%2)
        b = math.pi*a
        f = 1/Fc.mbf(b)
        return [f*Fc.mbf(b*math.sqrt(1-(((2*i)/(N-1)-1)**2))) for i in range(N)]

    #dolph chebyshev window
    #a: height ratio of main lobe to side lobe
    def dolphchebyshev(N,a=2):
        N = N + (1-N%2)
        k = int((N-1)/2)
        s = 10**a
        t = math.cosh((1/(N-1))*math.acosh(s))
        def w(i):
            return (1/N)*(s+2*sum([Fc.cp(N-1,t*math.cos(math.pi*j/N))*math.cos(2*(math.pi*j/N)*(i-k)) for j in range(1,k)]))
        r = [w(i) for i in range(N)]
        return [v/r[k] for v in r]

    #dirichlet kernel
    #n: degree
    #factor: (True: max=1, False: max=(1+2*n))
    def dirichlet(N,n=1,factor=True):
        N = N + (1-N%2)
        k = int((N-1)/2)
        if factor:
            f = 1/(1+2*n)
        else:
            f = 1
        r = []
        for i in range(-k,0):
            x = math.pi*i/k
            r.append(f*math.sin((n+0.5)*x)/math.sin(x/2))
        return r + [f*(1+2*n)] + list(reversed(r))

    #fejer kernel
    #n: degree
    #factor: (True: max=1, False: max=n)
    def fejer(N,n=1,factor=True):
        N = N + (1-N%2)
        k = int((N-1)/2)
        if factor:
            f = 1/n
        else:
            f = 1
        r = []
        for i in range(-k,0):
            x = math.pi*i/k
            r.append(f*(1/n)*(1-math.cos(n*x))/(1-math.cos(x)))
        return r + [f*n] + list(reversed(r))

    #ricker wavelet (edge detection kernel)
    #s: standard deviation
    def ricker(N,s=None):
        N = N + (1-N%2)
        k = int((N-1)/2)
        if s == None:
            s = 4/(3*(math.pi**(0.50)))
        f = 2/(math.sqrt(3*s)*(math.pi**(0.25)))
        r = []
        for i in range(-k,k+1):
            t = (i/s)**2
            r.append(f*(1-t)*math.exp(-t/2))
        return r

    #exponential/poisson window
    #t: time constant
    def exponential(N,t=1):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [math.exp(-abs(i-k)/t) for i in range(N)]

    #lanczos window
    #a: parameter
    def lanczos(N,a=1):
        def sinc(x):
            t = math.pi*x
            return math.sin(t)/t
        N = N + (1-N%2)
        k = int((N-1)/2)
        r = []
        for i in range(0,k):
            x = i/k-1
            r.append(sinc(x)*sinc(x/a))
        return r + [1] + list(reversed(r))

    #uniform PMF (sum to 1 for any N)
    def uniform(N):
        N = N + (1-N%2)
        return [1/N]*N

    #triangular PMF (sum to 1 for any N)
    def triangular(N):
        lc = int(N/2) + N%2  #left center  lc == rc   if N is odd
        rc = int(N/2) + 1    #right center rc == lc+1 if N is even
        slope = 1/(lc*rc)
        r = [slope*i for i in range(1,lc+1)]
        return r + r[::-1][N%2:]

    #rademacher PMF (sum to 1 for any N)
    def rademacher(N):
        N = N + (1-N%2)
        k = int((N-1)/2)
        return [0]*(k-1) + [0.5,0,0.5] + [0]*(k-1)

    #benford PMF (sum to 1 for any N)
    def benford(N):
        N = N + (1-N%2)
        return [math.log(1+1/(i+1),N+1) for i in range(N)]

    #zipf PMF (sum to 1 for any N)
    #s: parameter
    #q: parameter
    def zipf(N,s,q=0):
        N = N + (1-N%2)
        f = 1/Fc.ghn(N,s,q)
        return [f*((i+q)**(-s)) for i in range(1,N+1)]

    #skellam PMF (sum to 1 for any N)
    def skellam(N,u1=1,u2=1):
        N = N + (1-N%2)
        k = int((N-1)/2)
        if u1 < 0 or u2 < 0:
            return None
        return [math.exp(-(u1+u2))*((u1/u2)**(i/2))*Fc.mbf(2*math.sqrt(u1*u2),abs(i)) for i in range(-k,k+1)]

    #beta binomial PMF (sum to 1 for any N)
    def betabinomial(N,a=1,b=1):
        N = N + (1-N%2)
        f = Fc.beta(a,b)
        return [Fc.ncr(N-1,i)*Fc.beta(i+a,N-1-i+b)/f for i in range(N)]

    #binomial PMF (sum to 1 for any N)
    #p: success probability
    def binomial(N,p=0.5):
        N = N + (1-N%2)
        if p < 0 or p > 1:
            return None
        q = 1-p
        return [Fc.ncr(N-1,i)*(p**i)*(q**(N-1-i)) for i in range(N)]

    #hypergeometric PMF (sum to 1 for any N)
    #n: population size
    #k: number of success states
    def hypergeometric(N,n,k):
        N = N + (1-N%2)
        if N-1 > n or k > n:
            return None
        s = max(0,N-1+k-n)
        e = min(N-1,k)
        z = max(0,N-1-(e-s))
        return [Fc.ncr(k,i)*Fc.ncr(n-k,N-1-i)/Fc.ncr(n,N-1) for i in range(s,e+1)] + [0]*z

    #negative binomial PMF (sum to 1 for inf N)
    #r: number of failures
    #p: success probability
    def nbinomial(N,r,p=0.5):
        N = N + (1-N%2)
        if r <= 0 or p < 0 or p > 1:
            return None
        q = 1-p
        return [Fc.ncr(i+r-1,i)*(p**i)*(q**r) for i in range(N)]

    #negative hypergeometric PMF (sum to 1 for inf N)
    #r: number of failures
    #k: number of success states
    def nhypergeometric(N,r,n,k):
        N = N + (1-N%2)
        if r <= 0 or N-1 > n or r > n-k or k > n:
            return None
        e = min(N-1,k)
        z = max(0,N-1-e)
        return [Fc.ncr(i+r-1,i)*Fc.ncr(n-r-i,k-i)/Fc.ncr(n,k) for i in range(0,e+1)] + [0]*z

    #geometric PMF (sum to 1 for inf N)
    #p: success probability
    def geometric(N,p=0.5):
        N = N + (1-N%2)
        if p < 0 or p > 1:
            return None
        q = 1-p
        return [p*(q**i) for i in range(N)]

    #poisson PMF (sum to 1 for inf N)
    #l: event rate
    def poisson(N,l):
        f = math.exp(-l)
        return [f*(l**i)/math.factorial(i) for i in range(N)]

    #borel PMF (sum to 1 for inf N)
    #u: arrival rate
    #k: jobs
    def borel(N,u,k):
        N = N + (1-N%2)
        if u < 0 or u > 1 or k < 1:
            return None
        return [(k/i)*math.exp(-u*i)*((u*i)**(i-k))/math.factorial(i-k) for i in range(k,k+N)]

    def test():
        def floatrange(start,stop,step):  #return range of floats
            return [start+step*i for i in range(int((stop-start)*int(1/step)+1))]
        N = 51
        rng1 = dict()
        rng1["sin"]             = range(0,10)
        rng1["cgaussian"]       = floatrange(1.0, 9.0, 1.0)
        rng1["tukey"]           = floatrange(0.0, 1.0, 0.1)
        rng1["plancktaper"]     = floatrange(0.0, 0.5, 0.1)
        rng1["kaiser"]          = floatrange(0.0, 2.0, 0.2)
        rng1["dolphchebyshev"]  = floatrange(2.0, 9.0, 1.0)
        rng1["dirichlet"]       = floatrange(1.0, 9.0, 1.0)
        rng1["fejer"]           = floatrange(1.0, 9.0, 1.0)
        rng1["ricker"]          = floatrange(1.0, 9.0, 1.0)
        rng1["exponential"]     = floatrange(1.0, 9.0, 1.0)
        rng1["lanczos"]         = floatrange(0.1, 2.0, 0.2)
        rng1["binomial"]        = floatrange(0.0, 1.0, 0.1)
        rng1["geometric"]       = floatrange(0.0, 1.0, 0.1)
        rng1["poisson"]         = range(0,51,5)
        rng2 = dict()
        rng2["gaussian"]        = [floatrange(0.1,2.1,0.2),[2,4,6,8]]
        rng2["zipf"]            = [floatrange(0.0,1.0,0.3),floatrange(0.0,1.0,0.3)]
        rng2["skellam"]         = [range(1,10,4),range(1,10,4)]
        rng2["betabinomial"]    = [range(1,21,5),range(1,21,5)]
        rng2["hypergeometric"]  = [[200],range(0,201,20)]
        rng2["nbinomial"]       = [[10],floatrange(0.0,1.0,0.1)]
        rng2["borel"]           = [floatrange(0.0,1.0,0.2),range(1,8,3)]
        rng3 = dict()
        rng3["nhypergeometric"] = [range(1,102,20),[200],range(0,100,33)]
        kd = list(Kernel.kernels().items())
        rows = 4
        cols = int((len(kd)+rows-1)/rows)
        f, ax = plt.subplots(rows, cols)
        for i in range(rows):
            for j in range(cols):
                ax[i,j].set_xlim(-1,N)
                ax[i,j].set_ylim(-0.1,1.1)
                ax[i,j].tick_params(axis="x", direction="in", pad=0, labelsize=8)
                ax[i,j].tick_params(axis="y", direction="in", pad=0, labelsize=8)
                if i*cols+j >= len(kd):
                    continue
                name = kd[i*cols+j][0]
                func = kd[i*cols+j][1]
                ax[i,j].set_title(name, fontsize=8)
                if name == "cos":
                    for type in ["hann","hamming","blackman","nuttall","blackman-nuttall","blackman-harris","flattop"]:
                        y = func(N,type=type)
                        ax[i,j].plot(np.arange(len(y)),y)
                elif name in rng1.keys():
                    for a in rng1[name]:
                        y = func(N,a)
                        ax[i,j].plot(np.arange(len(y)),y)
                elif name in rng2.keys():
                    for a in rng2[name][0]:
                        for b in rng2[name][1]:
                            y = func(N,a,b)
                            ax[i,j].plot(np.arange(len(y)),y)
                elif name in rng3.keys():
                    for a in rng3[name][0]:
                        for b in rng3[name][1]:
                            for c in rng3[name][2]:
                                y = func(N,a,b,c)
                                ax[i,j].plot(np.arange(len(y)),y)
                else:
                    y = func(N)
                    ax[i,j].plot(np.arange(len(y)),y)
        plt.show()
        plt.close()

class Local():
    def differential(x):
        r = [0]*(len(x)-1)
        for i in range(1,len(x)):
            r[i-1] = x[i] - x[i-1]
        return r

    def differential2(x):
        r = [0]*(len(x)-2)
        for i in range(1,len(x)-1):
            r[i-1] = x[i+1] + x[i-1] - 2*x[i]
        return r

    def integral(x):
        r = [0]*len(x)
        for i in range(1,len(x)):
            r[i] = r[i-1] + (x[i] + x[i-1])/2
        return r

    def locals(x):
        r = Local.differential(x)
        i = 0
        trou_index = []
        trou_value = []
        peak_index = []
        peak_value = []
        while i < len(r):
            if r[i] < 0:
                while r[i] <= 0:
                    i = i + 1
                    if i >= len(r):
                        return dict(min_index=trou_index,min=trou_value,max_index=peak_index,max=peak_value)
                trou_index.append(i)
                trou_value.append(x[i])
            elif r[i] > 0:
                while r[i] >= 0:
                    i = i + 1
                    if i >= len(r):
                        return dict(min_index=trou_index,min=trou_value,max_index=peak_index,max=peak_value)
                peak_index.append(i)
                peak_value.append(x[i])
            else:
                while r[i] == 0:
                    i = i + 1
                    if i >= len(r):
                        return dict(min_index=trou_index,min=trou_value,max_index=peak_index,max=peak_value)
        return dict(min_index=trou_index,min=trou_value,max_index=peak_index,max=peak_value)

    def _lessthan(a,b):
        return a<b

    def _greaterthan(a,b):
        return a>b

    #flag indices with flat areas (adjacent duplicates)
    def flat(x):
        flags = [False]*len(x)
        i = 0
        while i < len(x)-1:
            if x[i] == x[i+1]:
                flags[i] = True
                flags[i+1] = True
            i = i + 1
        return flags

    #flag indices with min/max
    def _flagminmax(x,lookahead,cf,flags):
        for i in range(0,len(x)):
            if flags[i] != True:  #not a local min/max
                continue
            for j in range(i-lookahead,i+lookahead+1):
                if (j < 0) or (j > len(x)-1) or (j == i):  #out of bounds or same point
                    continue
                if cf(x[j],x[i]):
                    flags[i] = False
                    break
        return flags

    #flag indices with min/max with threshold
    #for local max: left avg < thr*x[i] and right avg < thr*x[i]
    #for local min: left avg > thr*x[i] and right avg > thr*x[i]
    def _flagminmaxthr(x,lookahead,thr,cf,flags):
        for i in range(0,len(x)):
            if flags[i] != True:  #not a local min/max
                continue
            #check left values
            left = []
            for j in range(i-lookahead,i):
                if (j < 0):
                    continue
                left.append(x[j])
            lpass = True
            if len(left) > 0:
                lavg = np.mean(left)
                if cf(lavg,thr*x[i]):
                    lpass = False
            #check right values
            right = []
            for j in range(i+1,i+lookahead+1):
                if (j > len(x)-1):
                    continue
                right.append(x[j])
            rpass = True
            if len(right) > 0:
                ravg = np.mean(right)
                if cf(ravg,thr*x[i]):
                    rpass = False
            #update flag
            flags[i] = lpass and rpass
        return flags

    #find local min/max (cf: comparison function - Local._lessthan to find min; Local._greaterthan to find max)
    def _minmax(x,lookahead,thr,cf):
        flags = [True]*len(x)
        Local._flagminmax(x,1,cf,flags)              #find all local min/max
        if lookahead > 1:                             #for all local min/max check lookahead values
            Local._flagminmax(x,lookahead,cf,flags)
        if thr != None:
            Local._flagminmaxthr(x,lookahead,thr,cf,flags)
        return flags

    #local minimum with look ahead
    #lookahead: number of values to look before/ahead of x[i] (lookahead >= 1)
    #thr: threshold ratio - for x[i] to be considered a minumum, avg(left pts) and avg(right pts) must be at least thr*x[i] (1.0<=thr)
    def min(x,lookahead,thr=None):
        return Local._minmax(x,lookahead,thr,Local._lessthan)

    #local maximum with look ahead
    #lookahead: number of values to look before/ahead of x[i] (lookahead >= 1)
    #thr: threshold ratio - for x[i] to be considered a maximum, avg(left pts) and avg(right pts) must be at most thr*x[i] (0.0<=thr<=1.0)
    def max(x,lookahead,thr=None):
        return Local._minmax(x,lookahead,thr,Local._greaterthan)

    #local differences (left-right) with lookahead
    #f: np.mean or np.median
    #percentage: if True return percentage difference
    def diff(x,lookahead,f,percentage):
        r = []
        for i in range(0,lookahead):
            r.append(0.0)
        for i in range(lookahead,len(x)-lookahead+1):
            left = f(x[i-lookahead:i])
            right = f(x[i:i+lookahead])  #include ith point
            if percentage:
                avg = (left+right)/2
                r.append(abs(left-right)/avg)
            else:
                r.append(abs(left-right))
        for i in range(0,lookahead):
            r.append(0.0)
        return r

    #return set of indices [starting point, ending point] of increasing slope
    #increasing slope [1,2,3,4,3,2,1] > [0,3]
    def inc_slope(x):
        slopes = []
        i = 0
        while i < len(x)-1:
            if x[i] <= x[i+1]:
                s = i
                i = i + 1
                while i < len(x)-1 and x[i] <= x[i+1]:
                    i = i + 1
                e = i
                if x[s] != x[e]:  #not a slope
                    slopes.append([s,e,x[e]-x[s]])
            else:
                i = i + 1
        return slopes

    #return set of indices [starting point, ending point] of decreasing slope
    #decreasing slope [1,2,3,4,3,2,1] > [3,6]
    def dec_slope(x):
        slopes = []
        i = 0
        while i < len(x)-1:
            if x[i] >= x[i+1]:
                s = i
                i = i + 1
                while i < len(x)-1 and x[i] >= x[i+1]:
                    i = i + 1
                e = i
                if x[s] != x[e]:  #not a slope
                    slopes.append([s,e,x[s]-x[e]])
            else:
                i = i + 1
        return slopes

#matrix (m=row,n=col)
class Mat():
    #symmetrize square matrix
    def sym(x):
        m = len(x)
        n = len(x[0])
        if m != n:
            print("error: m != n")
            return None
        r = [[None for y in range(m)] for x in range(n)]
        for i in range(0,m):
            r[i][i] = x[i][i]
        for i in range(0,m):
            for j in range(i+1,m):
                r[i][j] = x[i][j]
                r[j][i] = r[i][j]
        return r

    #transpose matrix
    def transpose(x):
        m = len(x)
        n = len(x[0])
        r = [[None for y in range(m)] for x in range(n)]
        for i in range(m):
            for j in range(n):
                r[j][i] = x[i][j]
        return r

    #x = [[1,1,2,2],
    #     [1,1,2,2],
    #     [3,3,4,5],
    #     [3,3,6,7]]
    #Mat.condense(x) = [[1, 2, 2],
    #                   [3, 4, 5],
    #                   [3, 6, 7]]
    #x = [[1,1,2,2,2],
    #     [1,1,2,2,2],
    #     [3,3,4,4,4],
    #     [3,3,4,4,4],
    #     [3,3,4,4,4]]
    #Mat.condense(x) = [[1, 2],
    #                   [3, 4]]
    def condense(x):
        #condense row
        m = len(x)
        n = len(x[0])
        xr = [x[0]]
        for i in range(m-1):
            ne = False
            for j in range(n):
                if x[i][j] != x[i+1][j]:
                    ne = True
                    break
            if ne:
                xr.append(x[i+1])
        #condense col
        m = len(xr)
        n = len(xr[0])
        xc = [[xr[i][0]] for i in range(m)]
        for j in range(n-1):
            ne = False
            for i in range(m):
                if xr[i][j] != xr[i][j+1]:
                    ne = True
                    break
            if ne:
                for i in range(m):
                    xc[i].append(xr[i][j+1])
        return xc

    #build a matrix from the difference of two vectors a and b
    def dvec(a,b,usenp=True):
        if len(a) != len(b):
            return None
        if usenp:
            return np.abs(np.subtract.outer(a,b))
        else:
            n = len(a)
            r = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(0,n):
                for j in range(i+1,n):
                    r[i][j] = abs(a[i]-b[j])
            return Mat.sym(r)

    #return a list of column sums ignoring diagonal
    def sumcol(x,usenp=True):
        n = len(x[0])
        if usenp:
            x = np.array(x)
            np.fill_diagonal(x, 0)
            r = np.sum(x, 0)
        else:
            r = [0 for _ in range(n)]
            r[0] = sum(x[0][1:n])
            r[n-1] = sum(x[n-1][0:n-1])
            for i in range(n):
                r[i] = sum(x[i][0:i])+sum(x[i][i+1:n])
                #r[i] = r[i]/n
        return r

    #return a list of column mins ignoring diagonal
    def mincol(x,usenp=True):
        n = len(x[0])
        if usenp:
            x = np.array(x)
            np.fill_diagonal(x, np.max(x))
            r = np.min(x, 0)
        else:
            r = [0 for _ in range(n)]
            r[0] = min(x[0][1:n])
            r[n-1] = min(x[n-1][0:n-1])
            for i in range(1,n-1):
                r[i] = min(min(x[i][0:i]),min(x[i][i+1:n]))
        return r

class MatrixProfile():
    #return the matrix profile (matrix)
    #y: sequence
    #n: length of subsequence
    #matrix_type: matrix type
    #profile_type: profile type
    #return_type: "mph": matrix,profile,histogram, "ph": profile,histogram
    #return matrix
    def mp(y,n,matrix_type="mean",profile_type="sum",return_type="mph"):
        #check inputs
        if matrix_type not in ["mean","var","lrdiff","median","maxdist","mid","euclidean","sim"]:
            print("invalid matrix_type:",matrix_type)
            return None
        if profile_type not in ["sum","min"]:
            print("invalid profile_type:",profile_type)
            return None
        if return_type not in ["mph","ph"]:
            print("invalid return_type:",return_type)
            return None
        y = np.array(y)
        m = None
        p = None
        h = None
        rwd = RollingWindow.dictionary()
        if matrix_type in rwd:
            q = rwd[matrix_type](y,n,endpoints="contract")
            if profile_type == "sum":
                p = Arr.sad(q)
            elif profile_type == "min":
                q = np.array(q)
                mask = np.ones(q.shape, dtype=bool)
                p = [0]*len(q)
                for i in range(len(q)):
                    mask[i] = False
                    p[i] = np.min(np.abs(q-q[i])[mask])
                    mask[i] = True
            h = RollingWindow.mean(p,n,endpoints="expand")
            if return_type == "mph":
                m = Mat.dvec(q,q)
                return m,p,h
            elif return_type == "ph":
                return p,h
        else:
            if matrix_type == "euclidean":
                size = len(y)-n+1
                m = [[0 for _ in range(size)] for _ in range(size)]
                for i in range(0,size):
                    for j in range(i+1,size):
                        m[i][j] = np.linalg.norm(y[i:i+n]-y[j:j+n])
                m = Mat.sym(m)
            elif matrix_type == "sim":
                size = len(y)-n+1
                m = []
                for i in range(0,size):
                    k = y[i:i+n]
                    c = Convolution.apply(y,k,endpoints="contract",normalize=False)
                    m.append([abs(v) for v in c])
                m = Mat.sym(m)
            else:
                print("invalid matrix_type:",matrix_type)
                return None
            if profile_type == "sum":
                p = Mat.sumcol(m)
            elif profile_type == "min":
                p = Mat.mincol(m)
            h = RollingWindow.mean(p,n,endpoints="expand")
            if return_type == "mph":
                return m,p,h
            elif return_type == "ph":
                return p,h

    #return anomaly flags
    #h: histogram
    #n: length of subsequence
    #w: elbow weight
    #return threshold, anomaly flags
    def detect_anomalies(h,n,w=5):
        _,thr = Elbow.search(sorted(h),n=1,weight=w,debug=False)
        size = len(h)
        flags = [0]*size
        for i in range(size):
            flags[i] = int(h[i] > thr)
        return thr, flags

    #extract matrix, profile, histogram, threshold, anomaly flags
    #y: sequence
    #n: length of subsequence
    #elbow_weight: elbow weight for setting threshold
    #matrix_type: matrix type
    #profile_type: profile type
    #return_type: "mph": matrix,profile,histogram, "ph": profile,histogram
    def extract(y,n,elbow_weight,matrix_type,profile_type,return_type):
        if return_type == "mph":
            m,p,h = MatrixProfile.mp(y,n,matrix_type=matrix_type,profile_type=profile_type,return_type=return_type)
            thr, flags = MatrixProfile.detect_anomalies(h,n,w=elbow_weight)
            return m,p,h,thr,flags
        else:
            p,h = MatrixProfile.mp(y,n,matrix_type=matrix_type,profile_type=profile_type,return_type=return_type)
            thr, flags = MatrixProfile.detect_anomalies(h,n,w=elbow_weight)
            return p,h,thr,flags

    #plot matrix
    #maintitle: main title
    #subtitles: list of subtitles
    #ms: list of matrices
    #savepath: path to save to
    #show: show plot
    def plot(maintitle,subtitles,ms,savepath=None,show=False):
        fig, axes = plt.subplots(1,len(ms),figsize=(10,6))
        fig.suptitle(maintitle)
        ax = axes.flat
        for i in range(len(ax)):
            ax[i].set_title(subtitles[i])
            ax[i].imshow(ms[i], cmap='viridis')
        if savepath != None:
            fig.savefig(savepath)
        if show:
            plt.show()
        plt.close()

    #short time fourier transform
    #n: interval
    def stft(x,n):
        _,c,_ = Arr.lcr(x,n)
        x = np.array([(np.fft.fft(v)*np.conjugate(np.fft.fft(v))).real for v in c])
        x = x/np.max(x, axis=0)  #normalize columns
        x = x.T
        x = x/np.max(x, axis=0)  #normalize rows
        return x

    #recurrence plot
    #eps: max difference threshold
    def rplot(x,eps=0.1):
        n = len(x)
        r = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            r[i][i] = 1
        for i in range(n):
            for j in range(i+1,n):
                if abs(x[i]-x[j]) <= eps:
                    r[i][j] = 1
                else:
                    r[i][j] = 0
        return Mat.sym(r)

    def test():
        #matrix profile test
        x1,y1 = Points.periodic_sin(5000,a=1,p=5000/100,noise_level=0.1)
        x2,y2 = Points.periodic_sin(2000,a=2,p=2000/100,noise_level=0.2)
        y = y1 + y2
        n = 99
        y_trend, y_detrend, y_seasonal, y_irregular = RollingWindow.decompose(y,trend_interval=n*2,trend_model="add",trend_center="mean",seasonal_interval=n*2,seasonal_model="add",seasonal_center="mean")
        types = ["mean","var","lrdiff"]
        #types = ["median","maxdist","mid"]
        #types = ["euclidean","sim"]
        m = [0]*len(types)
        p = [0]*len(types)
        h = [0]*len(types)
        thr = [0]*len(types)
        flags = [0]*len(types)
        for (i,type) in enumerate(types):
            start_time = time.time()
            #m[i],p[i],h[i],thr[i],flags[i] = MatrixProfile.extract(y,n=n,elbow_weight=1,matrix_type=type,profile_type="sum",return_type="mph")
            p[i],h[i],thr[i],flags[i] = MatrixProfile.extract(y,n=n,elbow_weight=1,matrix_type=type,profile_type="sum",return_type="ph")
            end_time = time.time()
            print("matrix profile",type,"seconds:",end_time - start_time)
        #MatrixProfile.plot("test",types,m,show=True)
        fig, axes = plt.subplots(len(types)+3,sharex=True,figsize=(10,6))
        x = np.arange(len(y))
        axes[0].plot(x,y)
        axes[1].plot(x,y_detrend)
        dy = np.abs([0]+Local.differential(y))
        _,hdy,_,_ = MatrixProfile.extract(dy,n=n,elbow_weight=1,matrix_type="mean",profile_type="sum",return_type="ph")
        axes[2].plot(x,hdy)
        for i in range(len(types)):
            axes[i+3].set_ylabel(types[i])
            axes[i+3].plot(x,h[i])
            axes[i+3].plot(x,sorted(h[i]))
            axes[i+3].plot(x,[thr[i]]*len(y))
            axes[i+3].plot(x,flags[i])
        plt.show()
        plt.close()
        #stft rplot test
        pi = math.pi
        f=200
        a = [math.cos(pi*6.25*i/f)+np.random.rand()*0.1 for i in range(0*f,1*f)]
        b = [math.cos(pi*12.5*i/f)+np.random.rand()*0.1 for i in range(1*f,2*f)]
        c = [math.cos(pi*25.0*i/f)+np.random.rand()*0.1 for i in range(2*f,3*f)]
        d = [math.cos(pi*50.0*i/f)+np.random.rand()*0.1 for i in range(3*f,4*f)]
        e = [0]*f
        y = a+b+c+d+e
        x1 = MatrixProfile.stft(y,100)
        x2 = MatrixProfile.rplot(y,0.5)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("Short Time Fourier Transform")
        im1 = ax1.imshow(x1, cmap="viridis", aspect="auto", origin="lower")
        fig.colorbar(im1, ax=ax1, pad=-0.15, shrink=0.9)
        ax2 = fig.add_subplot(312,sharex=ax1)
        ax2.set_title("Recurrence Plot")
        im2 = ax2.imshow(x2, cmap="viridis", aspect="auto", origin="lower")
        ax3 = fig.add_subplot(313,sharex=ax1)
        ax3.plot(np.arange(len(y)),y)
        plt.show()
        plt.close()

#todo: symmetry/skew mad
class Metrics():
    def dictionary():
        d = OrderedDict()
        #interval
        d["distance_min"] = Metrics.distance_min
        d["distance_max"] = Metrics.distance_max
        d["distance_center"] = Metrics.distance_center
        d["interval_size"] = Metrics.interval_size
        d["interval_intersect"] = Metrics.interval_intersect
        #work
        d["wasserstein"] = Metrics.wasserstein
        #edit
        d["lcs"] = Metrics.lcs
        d["levenshtein"] = Metrics.levenshtein
        d["jaccard"] = Metrics.jaccard
        #percent
        d["relative"] = Metrics.relative
        d["relative_log"] = Metrics.relative_log
        #statistical
        d["mean"] = Metrics.mean
        d["median"] = Metrics.median
        d["var"] = Metrics.var
        #distance
        d["manhattan"] = Metrics.manhattan  #l1
        d["euclidean"] = Metrics.euclidean  #l2
        d["chebyshev"] = Metrics.chebyshev  #linf
        #weighted l1
        d["sorensen"] = Metrics.sorensen
        d["kulczynski"] = Metrics.kulczynski
        d["soergel"] = Metrics.soergel
        d["canberra"] = Metrics.canberra
        d["lorentzian"] = Metrics.lorentzian
        #squared l2
        d["neyman"] = Metrics.neyman
        d["pearson"] = Metrics.pearson
        d["sqchi"] = Metrics.sqchi
        d["divergence"] = Metrics.divergence
        #wave
        d["cosine"] = Metrics.cosine
        d["correlation"] = Metrics.correlation
        d["covariance"] = Metrics.covariance
        d["dtw"] = Metrics.dtw
        #topology
        d["frechet"] = Metrics.frechet
        d["hausdoroff"] = Metrics.hausdoroff
        #probability
        d["bhattacharyya"] = Metrics.bhattacharyya
        d["hellinger"] = Metrics.hellinger
        #entropy
        d["kldivergence"] = Metrics.kldivergence
        d["jeffreys"] = Metrics.jeffreys
        d["kdivergence"] = Metrics.kdivergence
        d["topsoe"] = Metrics.topsoe
        d["jensen"] = Metrics.jensen
        #test
        d["differential"] = Metrics.differential
        d["peaks"] = Metrics.peaks
        d["area"] = Metrics.area
        d["angular"] = Metrics.angular
        d["centroid"] = Metrics.centroid
        d["convexhull"] = Metrics.convexhull
        return d

    def distance_min(a,b):
        return abs(min(a)-min(b))

    def distance_max(a,b):
        return abs(max(a)-max(b))

    def distance_center(a,b):
        return abs((max(a)-max(b))+(min(a)-min(b)))

    #return 0: small deviation in size
    #return 1: large deviation in size
    def interval_size(a,b):
        a_min = min(a)
        a_max = max(a)
        b_min = min(b)
        b_max = max(b)
        a_width = a_max-a_min
        b_width = b_max-b_min
        if a_width == 0 and b_width == 0:
            return 0
        elif a_width == 0 or b_width == 0:
            return 1
        else:
            return abs(a_width-b_width)/((a_width+b_width)/2)

    #jaccard distance for intersection
    #return 0: complete intersection
    #return 1: no intersection
    def interval_intersect(a,b):
        a_min = min(a)
        a_max = max(a)
        b_min = min(b)
        b_max = max(b)
        c_min =min([a_min,b_min])
        c_max = max([a_max,b_max])
        if a_min == b_min and a_max == b_max:    #complete intersection
            return 0
        elif a_min <= b_max and b_min <= a_max:  #intersection
            a_width = a_max-a_min
            b_width = b_max-b_min
            c_width = c_max-c_min
            return 2-(a_width+b_width)/c_width
        else:                                    #no intersection
            return 1

    #jaccard distance
    def jaccard(a,b):
        sum_min = Metrics._sum_min(a,b)
        sum_max = Metrics._sum_max(a,b)
        if sum_min == sum_max:
            return 0
        else:
            return 1-(sum_min/sum_max)

    def _scalingfactor(a,b,scl):
        if scl == "avg":
            return [(a[i]+b[i])/2 for i in range(len(a))]
        elif scl == "min":
            return [min(a[i],b[i]) for i in range(len(a))]
        elif scl == "max":
            return [max(a[i],b[i]) for i in range(len(a))]
        else:
            print("invalid scl:",scl)
            return None

    def _returnvalue(a,ret):
        if ret == "sum":
            return sum(a)
        elif ret == "avg":
            return np.mean(a)
        elif ret == "min":
            return min(a)
        elif ret == "max":
            return max(a)
        elif ret == "raw":
            return a
        else:
            print("invalid ret:",ret)
            return None

    #percent change
    def relative(a,b,scl="avg",ret="max"):
        scl_factor = Metrics._scalingfactor(a,b,scl)
        if scl_factor == None:
            return None
        p = [0]*len(a)
        for i in range(len(a)):
            if a[i] != b[i]:
                p[i] = abs(a[i]-b[i])/scl_factor[i]
        return Metrics._returnvalue(p,ret)

    #percent change
    def relative_log(a,b,ret="max"):
        p = [0]*len(a)
        for i in range(len(a)):
            if a[i] != b[i]:
                p[i] = abs(math.log(a[i])-math.log(b[i]))
        return Metrics._returnvalue(p,ret)

    def mean(a,b):
        return abs(np.mean(a)-np.mean(b))

    def median(a,b):
        return abs(np.median(a)-np.median(b))

    def var(a,b):
        return abs(np.var(a)-np.var(b))

    #L1 distance
    def manhattan(a,b):
        return Metrics.minkowski(a,b,1)

    #weighted L1
    def sorensen(a,b):
        return Metrics.manhattan(a,b)/(sum(a)+sum(b))

    def _sum_max(a,b):
        r = 0
        for i in range(len(a)):
            r = r + max(a[i],b[i])
        return r

    def _sum_min(a,b):
        r = 0
        for i in range(len(a)):
            r = r + min(a[i],b[i])
        return r

    #weighted L1
    def kulczynski(a,b):
        return Metrics.manhattan(a,b)/Metrics._sum_min(a,b)

    #weighted L1
    def soergel(a,b):
        return Metrics.manhattan(a,b)/Metrics._sum_max(a,b)

    #weighted L1
    def canberra(a,b):
        r = [0]*len(a)
        for i in range(len(a)):
            if a[i] != b[i]:
                r[i] = abs(a[i]-b[i])/(a[i]+b[i])
        return sum(r)

    #weighted L1
    def lorentzian(a,b):
        r = [0]*len(a)
        for i in range(len(a)):
            r[i] = math.log(1+abs(a[i]-b[i]))
        return sum(r)

    #L2 distance
    def euclidean(a,b):
        return Metrics.minkowski(a,b,2)

    #squared L2
    def neyman(a,b):
        sum = 0
        for i in range(len(a)):
            if a[i] != 0:
                sum = sum + (a[i]-b[i])*(a[i]-b[i])/a[i]
        return sum

    #squared L2
    def pearson(a,b):
        sum = 0
        for i in range(len(a)):
            if b[i] != 0:
                sum = sum + (a[i]-b[i])*(a[i]-b[i])/b[i]
        return sum

    #squared L2
    def sqchi(a,b):
        sum = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                sum = sum + (a[i]-b[i])*(a[i]-b[i])/(a[i]+b[i])
        return sum

    #squared L2
    def divergence(a,b):
        sum = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                x = (a[i]-b[i])*(a[i]-b[i])
                y = (a[i]+b[i])*(a[i]+b[i])
                sum = sum + x/y
        return sum

    #Linf distance
    def chebyshev(a,b):
        d_max = 0
        for i in range(len(a)):
            d = abs(a[i]-b[i])
            if d > d_max:
                d_max = d
        return d_max
        #return Metrics.minkowski(a,b,100)

    #Lp distance
    #p=1 manhattan
    #p=2 euclidean
    #p=inf chebyshev
    def minkowski(a,b,p=1):
        sum = 0
        for i in range(len(a)):
            sum = sum + math.pow(abs(a[i]-b[i]),p)
        return math.pow(sum,1/p)

    #entropy
    #defined for sum(p) = 1; sum(q) = 1; for all i if p[i] > 0 then q[i] > 0
    def kldivergence(a,b):
        a_p,b_p = Metrics._tobins(a,b,5)
        sum = 0
        for i in range(len(a_p)):
            if a_p[i] != 0 and b_p[i] != 0:  #lim[x to 0](x*log(x)) = 0
                sum = sum + a_p[i]*math.log(a_p[i]/b_p[i])
        return sum

    #entropy
    def jeffreys(a,b):
        sum = 0
        for i in range(len(a)):
            if a[i] != 0 and b[i] != 0:
                sum = sum + (a[i]-b[i])*math.log(a[i]/b[i])
        return sum

    #entropy
    def kdivergence(a,b):
        sum = 0
        for i in range(len(a)):
            if a[i] != 0 and b[i] != 0:
                sum = sum + a[i]*math.log(2*a[i]/(a[i]+b[i]))
        return sum

    #entropy
    def topsoe(a,b):
        sum = 0
        for i in range(len(a)):
            if a[i] != 0 and b[i] != 0:
                x = a[i]*math.log(2*a[i]/(a[i]+b[i]))
                y = b[i]*math.log(2*b[i]/(a[i]+b[i]))
                sum = sum + x + y
        return sum

    #entropy
    def jensen(a,b):
        a_p,b_p = Metrics._tobins(a,b,5)
        sum = 0
        for i in range(len(a_p)):
            if a_p[i] != 0 and b_p[i] != 0:
                x = a_p[i]*math.log(a_p[i])+b_p[i]*math.log(b_p[i])
                y = (a_p[i]+b_p[i])*math.log((a_p[i]+b_p[i])/2)
                sum = sum + x - y
        return sum

    def _dotproduct(a,b):
        sum = 0
        for i in range(len(a)):
            sum = sum + a[i]*b[i]
        return sum

    def harmonicmean(a,b):
        sum = 0
        for i in range(len(a)):
            sum = sum + a[i]*b[i]/(a[i]+b[i])
        return sum

    def dice(a,b):
        ab = Metrics._dotproduct(a,b)
        a2 = math.sqrt(Metrics._dotproduct(a,a))
        b2 = math.sqrt(Metrics._dotproduct(b,b))
        return 1-2*ab/(a2+b2)

    #cosine difference
    def cosine(a,b):
        ab = Metrics._dotproduct(a,b)
        a2 = math.sqrt(Metrics._dotproduct(a,a))
        b2 = math.sqrt(Metrics._dotproduct(b,b))
        a2b2 = a2*b2
        if a2b2 == 0:
            return 0
        else:
            return 1-ab/a2b2

    def _center(a):
        r = [0]*len(a)
        mean = np.mean(a)
        for i in range(len(a)):
            r[i] = a[i] - mean
        return r

    #centered cosine difference (pearson correlation coefficient)
    def correlation(a,b):
        a_centered = Metrics._center(a)
        b_centered = Metrics._center(b)
        return Metrics.cosine(a_centered,b_centered)

    def covariance(a,b):
        a_centered = Metrics._center(a)
        b_centered = Metrics._center(b)
        sum = 0
        for i in range(len(a)):
            sum = sum + a[i]*b_centered[i]
        return sum/len(a)

    #longest common subsequence
    def lcs(a,b,w=20,n=20):
        thr_min = min([min(a),min(b)])
        thr_max = max([max(a),max(b)])
        a_symbol = Metrics._symbol(a,thr_min,thr_max,n)
        b_symbol = Metrics._symbol(b,thr_min,thr_max,n)
        def update(r,a,b,i,j):
            if a[i-1] == b[j-1]:
                cost = 1
            else:
                cost = 0
            insert = r[j][i-1]
            delete = r[j-1][i]
            match = r[j-1][i-1] + cost
            r[j][i] = max(insert,delete,match)
        return 1 - Metrics._matrix(a_symbol,b_symbol,w=w,init_val=0,update=update)[len(b)][len(a)]/len(a)

    def levenshtein(a,b,w=20,n=20):
        thr_min = min([min(a),min(b)])
        thr_max = max([max(a),max(b)])
        a_symbol = Metrics._symbol(a,thr_min,thr_max,n)
        b_symbol = Metrics._symbol(b,thr_min,thr_max,n)
        def update(r,a,b,i,j):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1
            insert = r[j][i-1] + 1
            delete = r[j-1][i] + 1
            match = r[j-1][i-1] + cost
            r[j][i] = min(insert,delete,match)
        return Metrics._matrix(a_symbol,b_symbol,w=w,init_val=0,update=update)[len(b)][len(a)]/len(a)

    def _matrix(a,b,w,init_val,update):
        n = len(a)
        m = len(b)
        w = max(w,abs(n-m))  #abs(n-m)<=w
        r = [[init_val for x in range(n+1)] for y in range(m+1)]
        r[0][0] = 0
        for i in range(1,n+1):
            for j in range(max(1,i-w),min(m+1,i+w+1)):
                update(r,a,b,i,j)
        return r

    #dynamic time warping
    def dtw(a,b,w=3):
        def update(r,a,b,i,j):
            cost = abs(a[i-1]-b[j-1])
            insert = r[j][i-1]
            delete = r[j-1][i]
            match = r[j-1][i-1]
            r[j][i] = cost + min(insert,delete,match)
        return Metrics._matrix(a,b,w=w,init_val=float("inf"),update=update)[len(b)][len(a)]

    def frechet(a,b,w=3):
        def update(r,a,b,i,j):
            cost = abs(a[i-1]-b[j-1])
            insert = r[j][i-1]
            delete = r[j-1][i]
            match = r[j-1][i-1]
            r[j][i] = max(cost,min(insert,delete,match))
        return Metrics._matrix(a,b,w=w,init_val=float("inf"),update=update)[len(b)][len(a)]

    def hausdoroff(a,b):
        d_min = [0]*len(a)
        for i in range(len(a)):
            d = [0]*len(a)
            for j in range(len(a)):
                d[j] = abs(a[j]-b[i])
            d_min[i] = min(d)
        return max(d_min)

    #convert values to symbols
    #n: symbol set size
    def _symbol(a,thr_min,thr_max,n=10):
        r = [0]*(len(a))
        da = (thr_max-thr_min)/n  #bucket size
        if da != 0:
            for i in range(len(a)):
                symbol = int((a[i]-thr_min)/da)
                if symbol > n-1:
                    symbol = symbol-1
                r[i] = symbol
        return r

    #return a probability distribution by binning
    #n: odd number of buckets
    def _tobins(a,b,n):
        thr_min = min([min(a),min(b)])
        thr_max = max([max(a),max(b)])
        a_bin = Arr.bin(a,thr_min,thr_max,n)
        b_bin = Arr.bin(b,thr_min,thr_max,n)
        a_p = [v/len(a) for v in a_bin]
        b_p = [v/len(b) for v in b_bin]
        return a_p,b_p

    #return a probability distribution by exponential values
    def _softmax(a):
        e = [math.exp(v) for v in a]
        e_sum = sum(e)
        return [v/e_sum for v in e]

    #return a probability distribution by dividing by sum
    def _norm(a):
        a_sum = sum(a)
        return [v/a_sum for v in a]

    def _scale(x,a=0,b=1):
        xmin = min(x)
        xmax = max(x)
        if xmin == xmax:
            return [1 for v in x]
        return [(b-a)*(v-xmin)/(xmax-xmin)+a for v in x]

    #bhattacharyya coefficient
    #0 <= bc <= 1
    def _bc(a,b,n=5):
        if n==0:
            a_sum = sum(a)
            b_sum = sum(b)
            a_p = [v/a_sum for v in a]
            b_p = [v/b_sum for v in b]
        else:
            a_p,b_p = Metrics._tobins(a,b,n)
        bc = 0
        for i in range(len(a_p)):
            bc = bc + math.sqrt(a_p[i]*b_p[i])
        #float addition error propagation
        if bc > 1:
            bc = 1
        return bc

    #bhattacharyya distance
    #n: n bins for binning (if n=0 use probabilty distribution)
    def bhattacharyya(a,b,n=5):
        bc = Metrics._bc(a,b,n)
        if bc == 0:
            return 1
        else:
            return -1*math.log(bc)

    #hellinger distance
    #n: n bins for binning (if n=0 use probabilty distribution)
    def hellinger(a,b,n=5):
        bc = Metrics._bc(a,b,n)
        return math.sqrt(1-bc)

    #earth movers distance
    def wasserstein(a,b,weight=0.01):
        emd = [0]*(len(a)+1)
        for i in range(0,len(a)):
            emd[i+1] = emd[i] + a[i] - b[i]
        r = emd[len(emd)-1]
        remainder = abs(r)
        if r != 0:
            emd[len(emd)-1] = 0
            for i in range(len(emd)-2,-1,-1):
                if r*emd[i] <= 0:  #zero or opposite sign
                    break
                if abs(emd[i]) <= abs(r):
                    r = emd[i]
                    emd[i] = 0
                else:
                    emd[i] = emd[i] - r
        emd_sum = 0
        for i in range(len(emd)):
            emd_sum = emd_sum + abs(emd[i])
        return weight*emd_sum + remainder

    def _delta(a):
        r = [0]*(len(a)-1)
        for i in range(len(a)-1):
            r[i] = a[i+1]-a[i]
        return r

    def _signchange(a):
        r = [0]*len(a)
        for i in range(0,len(a)):
            if a[i] < 0:
                r[i] = -1
            elif a[i] > 0:
                r[i] = 1
        j = 1
        currentsign = r[0]
        while currentsign == 0 and j < len(r):
            currentsign = r[j]
            j = j + 1
        count = 0
        for i in range(j,len(r)):
            if r[i] != 0:
                if r[i] != currentsign:
                    currentsign = r[i]
                    count = count + 1
        return count

    def differential(a,b):
        a_delta = Metrics._delta(a)  #"length" of a
        b_delta = Metrics._delta(b)  #"length" of b
        return abs(np.mean(a_delta)-np.mean(b_delta))

    def peaks(a,b):
        a_peaks = int(Metrics._signchange(Metrics._delta(a))/2)
        b_peaks = int(Metrics._signchange(Metrics._delta(b))/2)
        return abs(a_peaks-b_peaks)

    def _shoelace(y):
        x = [0,1,2]
        return 0.5*abs(x[2]*(y[1]-y[0])-x[1]*(y[2]-y[0]))

    def area(a,b):
        a_area = 0
        b_area = 0
        for i in range(len(a)-2):
            a_area = a_area + Metrics._shoelace(a[i:i+3])
            b_area = b_area + Metrics._shoelace(b[i:i+3])
        return abs(a_area-b_area)

    #small dx > allow small difference between a and b
    #large dx > allow large difference between a and b
    def angular(a,b,dx=None):
        a_delta = Metrics._delta(a)
        b_delta = Metrics._delta(b)
        if dx == None:
            dx = np.mean(a_delta+b_delta)
        sum = 0
        for i in range(len(a_delta)):
            sum = sum + Metrics.cosine([a_delta[i],dx],[b_delta[i],dx])
        return sum/len(a_delta)

    def _centroid(a):
        xm = 0
        for i in range(len(a)):
            xm = xm + i*a[i]
        return xm/sum(a)

    def centroid(a,b):
        return abs(Metrics._centroid(a)-Metrics._centroid(b))

    def convexhull(a,b):
        a_2d = list(zip([i for i in range(len(a))],a))
        b_2d = list(zip([i for i in range(len(b))],b))
        c1 = ConvexHull.area(a_2d)
        c2 = ConvexHull.area(b_2d)
        if c1 == c2:
            return 0
        else:
            return 2*abs(c1-c2)/(c1+c2)

class Optics():
    #process data using optics clustering algorithm
    #x: data [[x1,y1],[x2,y2],...]
    #eps: distance to examine
    #minpts: minimum number of points within eps
    #return an ordered list of points
    def process(x, eps, minpts):
        orderedlist = []
        pts = []
        for i in range(len(x)):
            pts.append(Point(x[i][0],x[i][1],i))
        pts_tree = spatial.cKDTree(x)
        for p in pts:
            p.process(pts, pts_tree, eps, minpts)
        for p in pts:
            if p.processed == True:
                continue
            p.processed = True                      #mark p as processed
            orderedlist.append(p)                   #output p to orderedlist
            if p.core_distance != None:
                seeds = PriorityQueue()
                p.update(seeds)
                while not seeds.empty():
                    q = seeds.pop()
                    q.processed = True              #mark q as processed
                    orderedlist.append(q)           #output q to orderedlist
                    if q.core_distance != None:
                        q.update(seeds)
        return orderedlist

    #postprocess orderedlist - fill None values
    def fillnone(orderedlist):
        multiplier = 1.5
        #compute max values
        rds = [p.reachability_distance for p in orderedlist]
        cds = [p.core_distance for p in orderedlist]
        lrds = [p.local_reachability_density for p in orderedlist]
        rd_max = max(filter(lambda x: x is not None, rds))
        cd_max = max(filter(lambda x: x is not None, cds))
        lrd_max = max(filter(lambda x: x is not None, lrds))
        #fill None values with max values
        for p in orderedlist:
            if p.reachability_distance == None:        #new cluster or outlier
                p.reachability_distance = rd_max*multiplier
            if p.core_distance == None:                #outlier (neighbors < minpts)
                p.core_distance = cd_max*multiplier
            if p.local_reachability_density == None:   #infinite density (coinciding points)
                p.local_reachability_density = lrd_max*multiplier
        return orderedlist

    #return clusterids from orderedlist (use fillnone first)
    def dbscan_clusters(orderedlist, eps):
        clusterids = []
        clusterid = -1
        for p in orderedlist:
            if p.reachability_distance > eps:
                if p.core_distance <= eps:
                    clusterid = clusterid + 1
                    clusterids.append(clusterid)
                else:
                    clusterids.append(-1)
            else:
                clusterids.append(clusterid)
        return clusterids

    def test():
        #create data
        points = Points()
        n1 = points.add(Points.ring_uniform(50,50,30,60,5))
        n2 = points.add(Points.ring_random(0,0,5,7,100))
        n3 = points.add(Points.ring_random(50,50,0,20,100))
        n4 = points.add(Points.random(-10,150,-10,150,100))
        n5 = points.add(Points.random(-20,-10,-20,-10,20))

        #optics
        pts = points.zip()
        ordered = Optics.process(pts,eps=4000,minpts=5)
        ordered = Optics.fillnone(ordered)
        db_cid = Optics.dbscan_clusters(ordered,4)
        ordered_rd = [p.reachability_distance for p in ordered]
        ordered_cd = [p.core_distance for p in ordered]
        ordered_lrd = [p.local_reachability_density for p in ordered]
        ordered_idx = [p.index for p in ordered]

        inc = Local.inc_slope(ordered_rd)
        inc_rd = [0]*len(ordered_rd)
        for r in inc:
            s = r[0]
            e = r[1]
            for i in range(s,e+1):
                inc_rd[i]=r[2]

        dec = Local.dec_slope(ordered_rd)
        dec_rd = [0]*len(ordered_rd)
        for r in dec:
            s = r[0]
            e = r[1]
            for i in range(s,e+1):
                dec_rd[i]=r[2]

        fig = plt.figure()
        plt.axis('equal')
        ax = fig.add_subplot(111)
        ax.scatter(points.xs,points.ys,c=points.color())

        fig = plt.figure()
        plt.axis('equal')
        ax = fig.add_subplot(111)
        ax.scatter([p.x for p in ordered],[p.y for p in ordered],c=Points.idtocolor(db_cid))

        c = points.reorder(ordered_idx).color()
        Mplt.plots(np.arange(len(ordered_rd)),[ordered_rd,ordered_cd,ordered_lrd,inc_rd,dec_rd],[c for i in range(5)],fill=True,sharey=False)
        Mplt.show()
        Mplt.close()

class Outlier():
    #create replacement data for Outlier.replace
    #x: data
    #out: outlier result
    def rx_minmax(x,out):
        r = [0]*len(x)
        xmin = max(x)
        xmax = min(x)
        #calculate xmin and xmax with inliers
        for i in range(len(x)):
            if out[i] == 0:
                if x[i] < xmin:
                    xmin = x[i]
                elif x[i] > xmax:
                    xmax = x[i]
        #replace outliers with xmin and xmax
        for i in range(len(x)):
            if out[i] == 1:
                r[i] = xmax
            elif out[i] == -1:
                r[i] = xmin
        return r

    #replace outliers with inliers
    #x: data
    #out: outlier result
    #rx: replacement data
    def replace(x,out=None,rx=None):
        r = [0]*len(x)
        if out == None:
            out = Outlier.std(x,thr=2)
        if rx == None:
            rx = Outlier.rx_minmax(x,out)
        for i in range(len(x)):
            if out[i] == 0:
                r[i] = x[i]
            else:
                r[i] = rx[i]
        return r

    #detect outliers by bisection
    #x: data
    #thr: minimum number of points expected in half (1 < thr)
    #     for thr=1: since there is at least one point in each half when bisecting a list, will return no outlier
    #ratio: bisection ratio (0.0 < ratio < 1.0)
    #return 0 (inlier); 1 or -1 (outlier)
    def half(x,thr=3,ratio=0.5):
        r = [0]*len(x)
        x = list(x)  #local copy
        x_index = [i for i in range(len(x))]
        shift = 0
        while len(x) > 0:
            #calculate bisection point
            xmin = min(x)
            xmax = max(x)
            b = (xmax-xmin)*ratio+xmin
            #count the number of points above and below bisection point
            above_index = []
            above_count = 0
            below_index = []
            below_count = 0
            outlier_flag = [0]*len(x)
            for i in range(len(x)):
                if x[i] > b:
                    above_index.append(i)
                    above_count = above_count + 1
                    outlier_flag[i] = 1
                elif x[i] < b:
                    below_index.append(i)
                    below_count = below_count + 1
                    outlier_flag[i] = -1
            if (above_count == 0) and (below_count == 0):      #no points
                break
            if (above_count >= thr) and (below_count >= thr):  #no outlier points
                break
            #flag outliers
            outlier_index = []
            if above_count < thr:                        #number of points above bisection point that is below threshold
                outlier_index = outlier_index + above_index
            if below_count < thr:                        #number of points below bisection point that is below threshold
                outlier_index = outlier_index + below_index
            for i in reversed(sorted(outlier_index)):    #delete by last index
                r[x_index[i]] = outlier_flag[i]
                del x[i]
                del x_index[i]
        return r

    #return threshold range
    #x: data
    #thr: threshold factor
    #type: type of threshold
    def threshold(x,thr,type="std"):
        if type == "std":
            u = np.mean(x)
            std = np.std(x)
            thr_lo = u-thr*std
            thr_hi = u+thr*std
            return thr_lo,thr_hi
        elif type == "mad":
            v = np.median(x)
            mad = Dispersion.median_abs_dev(x)
            thr_lo = v-thr*mad
            thr_hi = v+thr*mad
            return thr_lo,thr_hi
        elif type == "iqr":
            thr_lo,thr_hi = Dispersion.iqrange(x,k=thr)
            return thr_lo,thr_hi

    #apply threshold range and return outlier flags
    def apply(x,thr_lo,thr_hi):
        r = [0]*len(x)
        for i in range(len(x)):
            if x[i] < thr_lo:
                r[i] = -1
            elif x[i] > thr_hi:
                r[i] = 1
        return r

    #detect outliers by standard deviation
    #x: data
    #thr: threshold
    #return 0 (inlier); 1 or -1 (outlier)
    def std(x,thr=2):
        thr_lo,thr_hi = Outlier.threshold(x,thr,"std")
        return Outlier.apply(x,thr_lo,thr_hi)

    #detect outliers by median absolute deviation
    #x: data
    #thr: threshold
    #return 0 (inlier); 1 or -1 (outlier)
    def mad(x,thr=2):
        thr_lo,thr_hi = Outlier.threshold(x,thr,"mad")
        return Outlier.apply(x,thr_lo,thr_hi)

    #detect outliers by tukeys fences
    #x: data
    #thr: threshold
    #return 0 (inlier); 1 or -1 (outlier)
    def iqr(x,thr=3):
        thr_lo,thr_hi = Outlier.threshold(x,thr,"iqr")
        return Outlier.apply(x,thr_lo,thr_hi)

    def test():
        x = [-10,-5,-3,-2,-1,0,0,0,1,2,3,5,10]
        print(Outlier.std(x,thr=2))
        print(Outlier.mad(x,thr=2))
        print(Outlier.iqr(x,thr=1.5))

class Partition():
    #bin array to matrix
    #example:
    # bin_array([0.2,0.4,0.6],5,0) = [0,1,0,0,0,  0,0,1,0,0,  0,0,0,1,0]
    def bin_array(arr,n,padding,pmf_row=None,pmf_col=None,flat=True):
        #partition
        p = []
        for i in range(len(arr)):
            p.append(Partition.bin_value(arr[i],n=n,padding=padding))
        r1 = None
        if pmf_row != None:
            r1 = []
            for i in range(len(p)):
                r1.append(Partition.smooth(p[i],pmf_row))
        r2 = None
        if pmf_col != None:
            p_transpose = np.array(p).T.tolist()
            r2 = []
            for i in range(len(p_transpose)):
                r2.append(Partition.smooth(p_transpose[i],pmf_col))
            r2 = np.array(r2).T.tolist()
        r = None
        if pmf_row != None and pmf_col != None:
            r = []
            for i in range(len(arr)):
                r.append([r1[i][j] + r2[i][j] for j in range(len(r1[i]))])
        elif pmf_row != None:
            r = r1
        elif pmf_col != None:
            r = r2
        else:
            r = p
        if flat:
            r_flat = []
            for sublist in r:
                for item in sublist:
                    r_flat.append(item)
            return r_flat
        else:
            return r

    #bin value from [0,1] into one of n bins
    #SAX: symbolic aggregate approximation
    #for n = 10:
    # -0.20 <= x <= -0.11 pad
    # -0.10 <= x <= -0.01 pad
    #  0.00 <= x <=  0.09 bin01
    #  0.10 <= x <=  0.19 bin02
    #  0.20 <= x <=  0.29 bin03
    #  0.30 <= x <=  0.39 bin04
    #  0.40 <= x <=  0.49 bin05
    #  0.50 <= x <=  0.59 bin06
    #  0.60 <= x <=  0.69 bin07
    #  0.70 <= x <=  0.79 bin08
    #  0.80 <= x <=  0.89 bin09
    #  0.90 <= x <=  1.00 bin10
    #  1.01 <= x <=  1.10 pad
    #  1.11 <= x <=  1.20 pad
    #example:
    # bin_value(0.1,10) = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    def bin_value(v,n,padding=0):
        p = [0]*n
        p1 = [0]*padding
        p2 = [0]*padding
        if math.isinf(v):
            if v < 0:
                p1[0] = 1
            else:
                p2[-1] = 1
        elif v == 1.0:
            index = n-1
            p[index] = 1
        elif v > 1.0:
            x = v*n
            index = int(v*n) - n
            if (int(x)-x) == 0:
                index = index - 1
            if index > padding-1:
                index = padding-1
            p2[index] = 1
        elif v < 0.0:
            x = v*n
            index = int(v*n) + (padding-1)
            if (int(x)-x) == 0:
                index = index + 1
            if index < 0:
                index = 0
            p1[index] = 1
        else:
            index = int(v*n)
            p[index] = 1
        p = p1 + p + p2
        return p

    #apply pmf to partition
    def smooth(p,pmf):
        pmf = pmf[int(len(pmf)/2):]+[0]*len(p)
        r = [0]*len(p)
        for i in range(len(p)):
            for j in range(len(p)):
                r[j] = r[j] + p[i]*pmf[abs(i-j)]
        return [v/max(r) for v in r]

    def test():
        p = Partition.bin_value(0.1,10,padding=2)
        n = Partition.smooth(p,Kernel.binomial(7,0.5))
        for i in range(len(p)):
            print(p[i],n[i])
        for v in [i/20 for i in range(-5,26)]:
            print(v,"\t",Partition.bin_value(v,10,padding=3))

class PCA():
    #covariance matrix
    def _cov_matrix(x,y):
        u1 = np.mean(x)
        u2 = np.mean(y)
        x_c = [v-u1 for v in x]
        y_c = [v-u2 for v in y]
        n = len(x)
        a = sum([v*v for v in x_c])/n
        b = sum([v1*v2 for v1,v2 in zip(x_c,y_c)])/n
        c = b
        d = sum([v*v for v in y_c])/n
        s = [[a,b],
             [c,d]]
        return s

    #https://en.wikipedia.org/wiki/Eigenvalue_algorithm
    def _eigenvectors(s):
        #calculate eigenvalues
        #det|[[a-eval,b     ],|
        #   | [c     ,d-eval]]|
        #eval*eval-(a+d)*eval+(a*d-b*c)=0
        tr = s[0][0]+s[1][1]                   #trace
        det = s[0][0]*s[1][1]-s[0][1]*s[1][0]  #determinant
        gap = math.sqrt(tr*tr-4*det)           #gap
        eval1 = (tr - gap)/2
        eval2 = (tr + gap)/2
        #calculate eigenvectors
        #(a-eval1)*v1+b*v2 = 0
        #c*v1+(d-eval1)*v2 = 0
        #evec = (a-eval,c) or (b,d-eval)
        m1 = math.sqrt(eval2*((s[0][0]-eval1)*(s[0][0]-eval1)+s[1][0]*s[1][0]))
        evec1 = ((s[0][0]-eval1)/m1,s[1][0]/m1)
        m2 = math.sqrt(eval1*((s[0][0]-eval2)*(s[0][0]-eval2)+s[1][0]*s[1][0]))
        evec2 = ((s[0][0]-eval2)/m2,s[1][0]/m2)
        return eval1,eval2,evec1,evec2

    #multiply matrix u by vector v
    def _mul(u,v):
        m = len(u)
        n = len(u[0])
        r = []
        for i in range(m):
            sum = 0
            for j in range(n):
                sum = sum + u[i][j]*v[j]
            r.append(sum)
        return r

    #change of basis
    def _cob(x,y,u):
        x_b = []
        y_b = []
        for v in list(zip(x,y)):
            t = PCA._mul(u,v)
            x_b.append(t[0])
            y_b.append(t[1])
        return x_b,y_b

    #http://ufldl.stanford.edu/wiki/index.php/PCA
    #http://ufldl.stanford.edu/wiki/index.php/Whitening
    def pca_whiten(x,y):
        u1 = np.mean(x)
        u2 = np.mean(y)
        x = [v-u1 for v in x]
        y = [v-u2 for v in y]
        s = PCA._cov_matrix(x,y)
        eval1,eval2,evec1,evec2 = PCA._eigenvectors(s)
        x_b,y_b = PCA._cob(x,y,[evec1,evec2])        #transform (x,y) to (x_b,y_b)
        return eval1,eval2,evec1,evec2,x_b,y_b

    def test():
        x1 = np.random.uniform(-2,0,size=10).tolist()
        y1 = np.random.uniform(-1,0,size=10).tolist()
        x2 = np.random.uniform(0,2,size=10).tolist()
        y2 = np.random.uniform(0,1,size=10).tolist()
        x = x1+x2
        y = y1+y2
        eval1,eval2,evec1,evec2,x_b,y_b = PCA.pca_whiten(x,y)
        plt.figure()
        ax = plt.subplot(211)
        plt.axis("equal")
        ax.scatter(x,y)
        u1 = np.mean(x)
        u2 = np.mean(y)
        ax.plot([u1,u1+evec1[0]],[u2,u2+evec1[1]],color="red")
        ax.plot([u1,u1+evec2[0]],[u2,u2+evec2[1]],color="black")
        ax = plt.subplot(212)
        plt.axis("equal")
        ax.scatter(x_b,y_b)
        plt.show()

#used for optics clustering
#self.reachability_distance == None > new cluster or outlier(neighbors < minpts)
#self.core_distance == None > outlier(neighbors < minpts)
class Point():
    def __init__(self,x,y,index):
        self.x = x
        self.y = y
        self.index = index
        self.neighbors = []
        self.core_distance = None
        self.reachability_distance = None
        self.processed = False  #reachability_distance processed flag
        self.local_reachability_density = None

    def __key__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__key__())

    #same memory location
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self is other
        return NotImplemented

    #comparison for priority queue
    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.reachability_distance < other.reachability_distance
        return NotImplemented

    #print
    def __str__(self):
        r = ""
        r = r + "point[" + str(self.index) + "] (" + str(self.x) + "," + str(self.y) + ")"
        r = r + "\n "
        r = r + "reachability=" + str(self.reachability_distance)
        r = r + "\n "
        r = r + "core=" + str(self.core_distance)
        r = r + "\n "
        r = r + "lrd=" + str(self.local_reachability_density)
        return r

    #process neighbors
    #pts: points array
    #pts_tree: spatial.cKDTree(x)
    #eps: distance
    #minpts: minimum number of points
    def process(self, pts, pts_tree, eps, minpts):
        self.neighbors = self.getneighbors(pts, pts_tree, eps)
        self.core_distance = self.coredistance(self.neighbors, minpts)
        self.reachability_distances = dict()
        if self.core_distance == None:  #neighbors < minpts
            for p in self.neighbors:
                self.reachability_distances[p] = self.distance(p)
        else:
            for p in self.neighbors:
                self.reachability_distances[p] = max(self.core_distance,self.distance(p))
        n_neighbors = len(self.neighbors)-1
        lrd_sum = sum(self.reachability_distances.values())
        if n_neighbors == 0:                                     #single point
            self.local_reachability_density = 0.0                #lowest density
        elif lrd_sum == 0:                                       #all points coincide
            self.local_reachability_density = None               #infinite density
        else:
            self.local_reachability_density = float(n_neighbors)/float(lrd_sum)
        self.centroid_point = self.centroid(self.getneighbors(pts, pts_tree, self.core_distance))

    def centroid(self,pts):
        x_sum = 0.0
        y_sum = 0.0
        for p in pts:
            x_sum = x_sum + p.x
            y_sum = y_sum + p.y
        n = len(pts)
        return Point(x_sum/n,y_sum/n,self.index)

    def distance(self,p):
        x = self.x-p.x
        y = self.y-p.y
        return math.sqrt(x*x+y*y)

    #return neighbors (sorted by nearest first)
    def getneighbors(self, pts, pts_tree, eps):
        indices = pts_tree.query_ball_point([self.x, self.y], eps)
        npts = [pts[i] for i in indices]
        #npts.remove(self)
        return npts

    def coredistance(self, npts, minpts):
        if len(npts) < minpts:
            return None
        else:
            distances = sorted([self.distance(p) for p in npts])
            return distances[minpts-1]

    #update seeds(priority queue)
    def update(self, seeds):
        for p in self.neighbors:
            if p.processed == False:
                 newreachdist = self.reachability_distances[p]
                 if p.reachability_distance == None:
                     p.reachability_distance = newreachdist
                     seeds.push(p)
                 else:
                      if newreachdist < p.reachability_distance:
                          p.reachability_distance = newreachdist
                          seeds.repush(p)

class Points():
    def __init__(self):
        self.id = -1
        self.xs = []
        self.ys = []
        self.ids = []

    #xy: tuple (xs,ys)
    def add(self,xy):
        self.id = self.id + 1
        xs,ys = xy
        self.xs = self.xs + xs
        self.ys = self.ys + ys
        self.ids = self.ids + [self.id for i in range(len(xs))]
        return len(xs)

    #reorder points
    def reorder(self,order):
        new_pts = Points()
        new_pts.id = self.id
        for i in order:
            new_pts.xs.append(self.xs[i])
            new_pts.ys.append(self.ys[i])
            new_pts.ids.append(self.ids[i])
        return new_pts

    #return colors
    def color(self):
        return Points.idtocolor(self.ids)

    #convert id to color
    def idtocolor(ids):
        n = len(ids)
        c = [(0.0, 0.0, 0.0, 1.0)]*n
        maxid = float(max(ids))
        for i in range(n):
            if ids[i] >= 0:
                c[i] = cm.hsv(float(ids[i])/maxid)
        return c

    def zip(self):
        return list(zip(self.xs,self.ys))

    #random points
    def random(x1,x2,y1,y2,n):
        x = list(np.random.uniform(x1,x2,n))
        y = list(np.random.uniform(y1,y2,n))
        return x,y

    #sin with noise
    #n: number of points
    #a: amplitude
    #p: number of periods
    #noise_level: noise level
    def periodic_sin(n,a=1,p=1,noise_level=1):
        x = np.linspace(0, 2*np.pi*p, n)
        noise = np.random.uniform(-1*noise_level,noise_level,n)
        y = [a*np.sin(x[i])+noise[i] for i in range(n)]
        return x,y

    #ring with uniformly distributed points
    #xi,yi: initial position
    #r1,r2: radius (0<=r1<=r2)
    #n: number of points
    def ring_random(xi,yi,r1,r2,n):
        u_r = np.random.uniform(0, 1, n)
        u_theta = np.random.uniform(0, 1, n)
        r1_sq = r1*r1
        r2_sq = r2*r2
        r = np.sqrt((r2_sq-r1_sq)*u_r+r1_sq);
        theta = 2*np.pi*u_theta
        x = [0]*n
        y = [0]*n
        for i in range(n):
            x[i] = xi + r[i] * np.cos(theta[i])
            y[i] = yi + r[i] * np.sin(theta[i])
        return x,y

    #ring with evenly spaced layers
    #xi,yi: initial position
    #r1,r2: radius (0<=r1<=r2)
    #n_r: number of layers
    #n_theta: number of points in layer
    def ring_spaced(xi,yi,r1,r2,n_r,n_theta):
        l_r = np.linspace(0,1,num=n_r)
        l_theta = np.linspace(0,1,num=n_theta,endpoint=False)
        r1_sq = r1*r1
        r2_sq = r2*r2
        r = np.sqrt((r2_sq-r1_sq)*l_r+r1_sq);
        theta = 2*np.pi*l_theta
        s_theta = theta[1]/2
        x = [0]*n_r*n_theta
        y = [0]*n_r*n_theta
        for i in range(n_r):          #layer
            for j in range(n_theta):  #point
                x[j+i*n_theta] = xi + r[i] * np.cos(theta[j]+s_theta*i)
                y[j+i*n_theta] = yi + r[i] * np.sin(theta[j]+s_theta*i)
        return x,y

    #ring with evenly spaced points
    #n_points = 6*(r1/d)*n_loops + 6*(1+2+3...) = 6*(r1/d)*n_loops + 6*(n_loops*(n_loops-1)/2) = 3*n_loops*(r1+r2)/d
    #xi,yi: initial position
    #r1,r2: radius (0<=r1<=r2 and divisible by d)
    #d: spacing distance
    def ring_uniform(xi,yi,r1,r2,d):
        if r1%d > 0:
            r1 = r1-(r1%d)+d
        if r2%d > 0:
            r2 = r2-(r2%d)
        if r1 == 0:
            r1 = d
            n_loops = int((r2-r1)/d+1)
            n_points = int(3*n_loops*(r1+r2)/d) + 1
            x = [0]*n_points
            y = [0]*n_points
            x[-1] = xi
            y[-1] = yi
        else:
            n_loops = int((r2-r1)/d+1)
            n_points = int(3*n_loops*(r1+r2)/d)
            x = [0]*n_points
            y = [0]*n_points
        c = 0
        for r in range(r1,r2+d,d):
            n_theta = int(6*r/d)
            l_theta = np.linspace(0,1,num=n_theta,endpoint=False)
            theta = 2*np.pi*l_theta
            for j in range(n_theta):
                x[c] = xi + r * np.cos(theta[j])
                y[c] = yi + r * np.sin(theta[j])
                c = c + 1
        return x,y

class Proximate():
    #check if value is near zero
    def zero(a,eps=1e-09):
        return (abs(a) <= eps)

    #check if two values a,b are near (float)
    def float(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    #check if two values a,b are close by percentage difference
    #diff/avg = percentage diff < factor
    #(warning when 0 is an element of (a,b)) let a = c (c not 0), b = 0 then 2 < factor: always False
    def isclose(a,b,factor=0.5):
        if a==b:
            return True
        return (2*abs(a-b) < factor*(a+b))

    #check if two values a,b are close by ratio
    #max/min = ratio < factor
    #(warning when 0 is an element of (a,b)) let a = c (c not 0), b = 0 then c < 0: always False
    def isclose2(a,b,factor=1.5):
        if a==b:
            return True
        return max(a,b) < min(a,b)*factor

class RollingWindow():
    def dictionary():
        d = OrderedDict()
        d["mean"] = RollingWindow.mean
        d["var"] = RollingWindow.var
        d["lrdiff"] = RollingWindow.lrdiff
        d["median"] = RollingWindow.median
        d["maxdist"] = RollingWindow.maxdist
        d["mid"] = RollingWindow.mid
        return d

    #cma(n) = [x(1)+...+x(n)]/(n)
    #cma(n+1) = [x(n+1)+n*cma(n)]/(n+1)
    def cumulative(x):
        cma = [0]*len(x)
        cma[0] = x[0]
        for i in range(1,len(x)):
            cma[i] = (x[i]+i*cma[i-1])/(i+1)
        return cma

    #ema(1) = x(1)
    #ema(n) = m*x(n)+(1-m)*ema(n-1)
    def exponential(x,m=0.05):
        ema = [0]*len(x)
        ema[0] = x[0]
        for i in range(1,len(x)):
            ema[i] = m*x[i]+(1-m)*ema[i-1]
        return ema

    #simple moving average
    #sma(n) = sma(n-1) + x(i)/n - x(i-n)/n
    #y: data
    #interval: window interval
    #endpoints: "expand" (expand endpoints) or "keep" (keep endpoints) or "contract" (contract endpoints)
    def mean(y,interval,endpoints="keep"):
        return Convolution.apply(y,Kernel.uniform(interval),endpoints=endpoints)

    #moving var
    #y: data
    #interval: window interval
    #endpoints: "keep" (keep endpoints) or "contract" (contract endpoints)
    def var(y,interval,endpoints="keep"):
        ey = Convolution.apply(y,Kernel.uniform(interval),endpoints=endpoints)
        e_y2 = Convolution.apply([v*v for v in y],Kernel.uniform(interval),endpoints=endpoints)
        ey_2 = [v*v for v in ey]
        return [e_y2[i]-ey_2[i] for i in range(len(ey))]

    #left/right differential
    #y: data
    #interval: window interval
    #endpoints: "keep" (keep endpoints) or "contract" (contract endpoints)
    def lrdiff(y,interval,endpoints="keep"):
        return Convolution.apply(y,Kernel.lrdiff(interval),endpoints=endpoints,normalize=False)

    #moving median
    #y: data
    #interval: window interval
    #endpoints: "keep" (keep endpoints) or "contract" (contract endpoints)
    def median(y,interval,endpoints="keep"):
        if endpoints == "keep":
            l,c,r = Arr.lcr(y,interval)
            lcr = [np.median(v) for v in l+c+r]
            return lcr
        elif endpoints == "contract":
            #reference: https://discuss.leetcode.com/topic/74634/easy-python-o-nk
            if isinstance(y, np.ndarray):
                y = y.tolist()
            window = sorted(y[:interval])
            r = [0]*(len(y)-interval+1)
            i = 0
            idx1 = int(interval/2)
            idx2 = -(idx1+1)
            for a, b in zip(y, y[interval:] + [0]):
                r[i] = (window[idx1] + window[idx2]) / 2.
                i = i + 1
                window.remove(a)
                bisect.insort(window, b)
            return r
            #return [np.median(y[i:i+interval]) for i in range(len(y)-interval+1)]

    #max distance
    #y: data
    #interval: window interval
    #endpoints: "keep" (keep endpoints) or "contract" (contract endpoints)
    def maxdist(y,interval,endpoints="keep"):
        def _maxdist(y):  #return the max distance
            r = 0
            temp = sorted(y)
            for i in range(len(y)-1):
                dist = abs(temp[i]-temp[i+1])
                if dist > r:
                    r = dist
            return r
        return RollingWindow.custom(y,interval,endpoints,_maxdist)

    #moving center based on min and max
    #y: data
    #interval: window interval
    #endpoints: "keep" (keep endpoints) or "contract" (contract endpoints)
    def mid(y,interval,endpoints="keep"):
        def _mid(y):  #return the mid point
            return (min(y)+max(y))/2
        return RollingWindow.custom(y,interval,endpoints,_mid)

    #custom moving window function
    #y: data
    #interval: window interval
    #endpoints: "keep" (keep endpoints) or "contract" (contract endpoints)
    #f: custom function
    def custom(y,interval,endpoints="keep",f=sum):
        if endpoints == "keep":
            l,c,r = Arr.lcr(y,interval)
            lcr = [f(v) for v in l+c+r]
            return lcr
        elif endpoints == "contract":
            return [f(y[i:i+interval]) for i in range(len(y)-interval+1)]

    def decompose_trend(y,interval,model="add",center="mean"):
        if center == "mean":
            y_trend = RollingWindow.mean(y,interval)
        elif center == "median":
            y_trend = RollingWindow.median(y,interval)
        if model == "add":
            y_detrend = [y[i]-y_trend[i] for i in range(len(y))]
        elif model == "mul":
            y_detrend = [y[i]/y_trend[i] for i in range(len(y))]
        return y_trend, y_detrend

    def decompose_seasonal(y_detrend,interval,model="add",center="mean"):
        n_detrend = len(y_detrend)
        y_singular = [[] for _ in range(interval)]
        for i in range(n_detrend):
            y_singular[i%interval].append(y_detrend[i])
        if center == "mean":
            y_center = [np.mean(arr) for arr in y_singular]
        elif center == "median":
            y_center = [np.median(arr) for arr in y_singular]
        y_seasonal = [0]*n_detrend
        for i in range(n_detrend):
            y_seasonal[i] = y_center[i%interval]
        if model == "add":
            y_irregular = [y_detrend[i]-y_seasonal[i] for i in range(n_detrend)]
        elif model == "mul":
            y_irregular = [y_detrend[i]/y_seasonal[i] for i in range(n_detrend)]
        return y_seasonal, y_irregular

    def decompose(y,trend_interval=None,trend_model="add",trend_center="mean",seasonal_interval=None,seasonal_model="add",seasonal_center="mean"):
        y_trend,y_detrend = RollingWindow.decompose_trend(y,interval=trend_interval,model=trend_model,center=trend_center)
        y_seasonal,y_irregular = RollingWindow.decompose_seasonal(y_detrend,interval=seasonal_interval,model=seasonal_model,center=seasonal_center)
        return y_trend, y_detrend, y_seasonal, y_irregular

    def test():
        y = [0]*20 + [1]*20 + [0]*20 + [1]*20 + [2]*20 + [1]*20 + [0]*20 + [i/10 for i in range(0,11,1)] + [i/10 for i in range(10,-1,-1)]
        intervals = [i for i in range(2,10)]
        shifts = [int(i/2) for i in intervals]
        y_mean = [[] for _ in range(len(intervals))]
        for i in range(len(intervals)):
            y_mean[i] = Convolution.apply(y,Kernel.ricker(intervals[i]),endpoints="keep",normalize=False)
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(np.arange(len(y)),y)
        linestyles = ["-","--","-.",":"]
        for i in range(len(intervals)):
            ax.plot(np.arange(len(y_mean[i])),y_mean[i],linestyle=linestyles[i%4],label=intervals[i])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.close()
        x = np.arange(100)
        y = [math.sin(v/10) for v in x]
        ys = []
        xs = []
        ts = ["y","mean","var","lrdiff","median","maxdist","mid","skew","kurt"]
        fig, axes = plt.subplots(9,sharex=True)
        [axes[i].set_ylabel(ts[i]) for i in range(9)]
        axes[0].scatter(np.arange(len(y)), y)
        m1 = RollingWindow.mean(y,25,endpoints="expand")
        m2 = RollingWindow.mean(y,25,endpoints="keep")
        m3 = RollingWindow.mean(y,25,endpoints="contract")
        axes[1].scatter(np.arange(-12,len(m1)-12), m1)
        axes[1].scatter(np.arange(len(m2)), m2)
        axes[1].scatter(np.arange(12,len(m3)+12), m3)
        i = 2
        for f in [RollingWindow.var, RollingWindow.lrdiff, RollingWindow.median, RollingWindow.maxdist, RollingWindow.mid]:
            s = f(y,25,endpoints="keep")
            d = f(y,25,endpoints="contract")
            axes[i].scatter(np.arange(len(s)), s)
            axes[i].scatter(np.arange(12,len(d)+12), d)
            i = i + 1
        t = RollingWindow.custom(y,25,f=Shape.skewness)
        axes[i].scatter(np.arange(len(t)), t)
        i = i + 1
        t = RollingWindow.custom(y,25,f=Shape.kurtosis)
        axes[i].scatter(np.arange(len(t)), t)
        i = i + 1
        plt.show()
        plt.close()

class Scaler():
    #scale list of values in x to [a,b]
    def scale(x,a=0,b=1,xmin=None,xmax=None,replaceoutliers=False):
        if replaceoutliers:
            x = Outlier.replace(x)
        if xmin == None:
            xmin = min(x)
        if xmax == None:
            xmax = max(x)
        if xmin == xmax:
            r = [a]*len(x)
            for i in range(len(x)):
                if x[i] < xmin:
                    r[i] = -1*float("inf")
                elif x[i] > xmin:
                    r[i] = float("inf")
            return r
            #return [0 for v in x]
        return [(b-a)*(v-xmin)/(xmax-xmin)+a for v in x]

    #scale list of values in x by max magnitude (will always produce positive values if all values are negative or positive)
    def scale_max(x):
        xmin = min(x)
        xmax = max(x)
        axmin = abs(xmin)
        axmax = abs(xmax)
        if axmin > axmax:
            div = xmin
        else:
            div = xmax
        if div == 0:
            return x,div
        else:
            return [v/div for v in x],div

    #scale list of values in x by div
    def scale_div(x,div):
        if div == 0:
            return x
        else:
            return [v/div for v in x]

    #convert to z-score
    def zscore(x):
        u = np.mean(x)
        std = np.std(x)
        return [(v-u)/std for v in x]

class Shape():
    #central moment (https://en.wikipedia.org/wiki/Central_moment)
    def _cm(x,n):
        u = np.mean(x)
        y = [0]*len(x)
        for i in range(len(x)):
            y[i] = (x[i]-u)**n
        return sum(y)/len(y)

    #standardized moment (https://en.wikipedia.org/wiki/Standardized_moment)
    def _sm(x,n):
        std = np.std(x)
        if std == 0:
            return 0
        else:
            return Shape._cm(x,n)/(std**n)

    #skewness
    #type: "m"  = moment coefficient of skewness (https://en.wikipedia.org/wiki/Skewness)
    #      "np" = nonparametric skewness (https://en.wikipedia.org/wiki/Nonparametric_skew)
    #      "q"  = quantile skewness
    def skewness(x,type="m"):
        if type == "m":
            return Shape._sm(x,3)
        elif type == "np":
            u = np.mean(x)
            v = np.median(x)
            std = np.std(x)
            if std == 0:
                return 0
            else:
                return (u-v)/std
        elif type == "q":
            q1 = Dispersion.percentile(x,0.25)
            q2 = Dispersion.percentile(x,0.50)
            q3 = Dispersion.percentile(x,0.75)
            iqr = q3-q1
            if iqr == 0:
                return 0
            else:
                return (q3+q1-2*q2)/iqr

    #kurtosis
    #type: "m"  = moment coefficient of kurtosis (https://en.wikipedia.org/wiki/Kurtosis)
    def kurtosis(x,type="m"):
        if type == "m":
            return Shape._sm(x,4)

class Spectrum():
    #autocorrelation
    def ac(x):
        fft = np.fft.fft(x)
        ifft = np.fft.ifft(fft*np.conj(fft))
        return ifft.real

    #extract amplitude and phase components
    def ap(x):
        fft = np.fft.fft(x,len(x))
        a = []  #amplitude
        p = []  #phase
        thr_real_p = max([abs(v) for v in fft.real])/10
        thr_imag_p = max([abs(v) for v in fft.imag])/10
        for i in range(len(fft.real)):  #same as np.sqrt(fft*np.conj(fft))
            m = math.sqrt(fft.real[i]*fft.real[i]+fft.imag[i]*fft.imag[i])
            a.append(m)
        for i in range(len(fft.real)):
            if abs(fft.real[i]) < thr_real_p or abs(fft.imag[i]) < thr_imag_p:
                p.append(0)
            else:
                p.append(math.atan2(fft.imag[i], fft.real[i])/np.pi)
        return np.fft.fftshift(a),np.fft.fftshift(p)

    #algorithm used to find peaks in magnitude spectrum
    #from https://stackoverflow.com/a/22640362/6029703
    #https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887
    def thresholding_algo(y, lag, threshold, influence):
        signals = np.zeros(len(y))
        filteredY = np.array(y)
        avgFilter = [0]*len(y)
        stdFilter = [0]*len(y)
        avgFilter[lag - 1] = np.mean(y[0:lag])
        stdFilter[lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(y)):
            if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
                if y[i] > avgFilter[i-1]:
                    signals[i] = 1
                else:
                    signals[i] = -1
                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.mean(filteredY[(i-lag):i])
                stdFilter[i] = np.std(filteredY[(i-lag):i])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = np.mean(filteredY[(i-lag):i])
                stdFilter[i] = np.std(filteredY[(i-lag):i])
        return np.asarray(signals), np.asarray(avgFilter), np.asarray(stdFilter)

    #returns:
    #t time array for ac
    #t_f time array for m, a, avgFilter, thr_upper, thr_lower, signals
    #ac autocorrelation array
    #a amplitude component
    #p phase component
    #thr_avg average filter for threshold
    #thr_upper upper threshold
    #thr_lower lower threshold
    #thr_signals threshold signals
    #flagonly return flag only
    #saveplot path to save plot
    #flag True when period n is found using threshold_stddev and threshold_rng
    def analyze(x, n=24, threshold_stddev=5, threshold_rng=0.5, flagonly=True, saveplot=None):
        x = Arr.nan_fill(x,fill="mean")
        x_trend, x_detrend = RollingWindow.decompose_trend(x,interval=n,model="add",center="mean")  #detrend
        x = Outlier.replace(x_detrend,out=None,rx=None)                                             #remove outliers
        t = np.arange(len(x))
        freq = np.fft.fftshift(np.fft.fftfreq(len(x)))
        ac = Spectrum.ac(x)
        a,p = Spectrum.ap(x)
        zero_index = int(len(freq)/2)
        a = a[zero_index+1:]
        p = p[zero_index+1:]
        freq = freq[zero_index+1:]
        t_f = 1/freq
        a,_ = Arr.zipsort(a,t_f)
        p,t_f = Arr.zipsort(p,t_f)
        signals, avgFilter, stdFilter = Spectrum.thresholding_algo(a,lag=n,threshold=threshold_stddev,influence=0)
        thr_upper = avgFilter + threshold_stddev * stdFilter
        thr_lower = avgFilter - threshold_stddev * stdFilter
        signals_sort,t_f_sort = Arr.zipsort(signals, t_f)
        i1,i2 = Arr.findrange(t_f_sort, n-threshold_rng, n+threshold_rng)
        flag = False
        for i in range(i1,i2+1):
            if signals_sort[i] == 1:
                flag = True
                break
        if saveplot != None:
            Spectrum.plot(x,t,t_f,ac,a,p,avgFilter,thr_upper,thr_lower,signals,flag,saveplot)
        if flagonly:
            return flag
        else:
            return x,t,t_f,ac,a,p,avgFilter,thr_upper,thr_lower,signals,flag

    #return x y for y not zero
    def nonzero(x,y):
        xr = []
        yr = []
        for i in range(len(x)):
            if y[i] != 0:
                xr.append(x[i])
                yr.append(0)
        return xr,yr

    def plot(x,t,t_f,ac,a,p,avgFilter,thr_upper,thr_lower,signals,flag,saveplot=None):
        fig, ax = plt.subplots(nrows=4,ncols=1)
        ax[0].plot(t,x)
        ax[1].set_xlim(0,50)
        ax[1].plot(t_f,a)
        ax[1].plot(t_f,avgFilter)
        ax[1].plot(t_f,thr_upper)
        ax[1].plot(t_f,thr_lower)
        x_sig,y_sig = Spectrum.nonzero(t_f,signals)
        ax[1].scatter(x_sig,y_sig,color="red")
        ax[2].set_xlim(0,50)
        ax[2].plot(t_f,p)
        ax[3].plot(t,ac)
        if saveplot == None:
            plt.show()
        else:
            plt.savefig(saveplot)
        plt.close(fig)

    def test():
        noise = np.random.uniform(-12,12,24*10)
        #x = [10]*2400 + [i%24-12 + noise[i] for i in range(24*10)]
        #x = [i%24-12 + noise[i] + 100 for i in range(24*10)] + [-100] + [i%24-12 + noise[i] for i in range(24*10)]
        _,x = Points.periodic_sin(24,noise_level=1)
        x = x*10
        #random walk
        """
        x = np.random.randint(2,size=100)*2-1
        c = 0
        for i in range(len(x)):
            c = c + x[i]
            x[i] = c
        """
        x,t,t_f,ac,a,p,avgFilter,thr_upper,thr_lower,signals,flag = Spectrum.analyze(x,flagonly=False,debug=True)

class Time():
    dayofweek = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    #precondition: fillandpartition
    def _subdivide(tsp,n=24):
        for i in range(len(tsp)):
            data = tsp[i].getdata()
            data = Arr.interpolate(data,1,Arr.inter_qspline)
            data = Arr.clip(data,minvalue=0)
            tsp[i].setdata(data)
            index_start,index_end = tsp[i].getindex()
            r = int((len(data)-1)/(n-1))
            tsp[i].setindex(index_start*r,index_start*r+len(data)-1)
        return tsp

    #get day of week
    def _weekday(t,asstring=True):
        t = DateTimeReader.todatetime(t)
        if asstring:
            return Time.dayofweek[t.weekday()]
        else:
            return t.weekday()

    #return day of week flags
    def _dayofweek(tlist):
        wkday_flag = [1]*len(tlist)
        wkend_flag = [0]*len(tlist)
        for i in range(len(tlist)):
            if Time._weekday(tlist[i]) in ["Saturday","Sunday"]:
                wkday_flag[i] = 0
                wkend_flag[i] = 1
        return wkday_flag,wkend_flag

    #return holiday flags
    def _holiday(tlist,hlist):
        hday_flag = [0]*len(tlist)
        t_dates = [DateTimeReader.todatetime(tstr).date() for tstr in tlist]
        h_dates = [datetime.datetime.strptime(hstr,'%Y-%m-%d').date() for hstr in hlist]
        for i in range(len(tlist)):
            if t_dates[i] in h_dates:
                hday_flag[i] = 1
        return hday_flag

    #return day flags
    #t: time array (["2000/01/01 1:00:00","2000/01/01 4:00:00","2000/01/01 5:00:00"])
    #h: holiday array (["2000-01-01"])
    def dayflags(t,h=None):
        #day of week flags
        wkday, wkend = Time._dayofweek(t)
        if h == None:
            hday = [0]*len(t)
        else:
            hday = Time._holiday(t,h)
        offday = [hday[i] or wkend[i] for i in range(len(t))]  #if holiday or weekend then offday
        return wkday,wkend,hday,offday

    #return hour difference between t1 and t2
    def _hourdifference(t1,t2):
        t1 = DateTimeReader.todatetime(t1)
        t2 = DateTimeReader.todatetime(t2)
        td = (t2-t1)
        return td.days*24+int(td.seconds/3600)

    #convert datetime to int (hour)
    def _converthour(x):
        if isinstance(x, list):
            return [int(DateTimeReader.todatetime(v).hour) for v in x]
        else:
            return int(DateTimeReader.todatetime(x).hour)

    #partition by 24 hour days
    def daypartition(t,y):
        tr = []
        yr = []
        #find first full day (having hours 0-23)
        for i in range(0,len(t),1):
            if Time._converthour(t[i]) == 0:
                s = i
                break
        #find last full day (having hours 0-23)
        for i in range(len(t)-1,-1,-1):
            if Time._converthour(t[i]) == 23:
                e = i
                break
        ndays = int((e-s+1)/24)
        for i in range(ndays):
            s1 = s+i*24
            e1 = s1+24
            tr.append(t[s1:e1])
            yr.append(y[s1:e1])
        return ndays,s,e,tr,yr

    #fill time series by datetime (skipped hours are filled in with np.nan)
    #example:
    # t=["2000/01/01 1:00:00","2000/01/01 4:00:00","2000/01/01 5:00:00"]
    # y=[100,200,300]
    # hr,tr,yr,fr = fill(t,y,v=np.nan,filltype=None)
    # hr = [1, 2, 3, 4, 5]
    # tr = ["2000/01/01 1:00:00","2000/01/01 2:00:00","2000/01/01 3:00:00","2000/01/01 4:00:00","2000/01/01 5:00:00"]
    # yr = [100, nan, nan, 200, 300]  #filltype=None
    # yr = [100, 150, 150, 200, 300]  #filltype="mean"
    # yr = [100, 0, 0, 200, 300]      #filltype=0
    # yr = [100, 1, 1, 200, 300]      #filltype=1
    # fr = [0, 1, 1, 0, 0]
    def fill(t,y,v=np.nan,filltype="mean"):
        hr = []
        tr = []
        yr = []
        fr = []
        i = 0
        if len(t) < 2:
            print("partition - partition undefined for 0 or 1 values")
            return None
        if len(t) != len(y):
            print("partition - partition undefined for different length arrays")
            return None
        while i < len(t)-1:
            hours = Time._hourdifference(t[i],t[i+1])
            currenthour = Time._converthour(t[i])
            currentdt = DateTimeReader.todatetime(t[i])
            hr.append(currenthour)
            tr.append(currentdt.strftime('%Y/%m/%d %H:%M:%S'))
            yr.append(y[i])
            fr.append(0)
            if hours != 1:
                for j in range(0,hours-1):
                    hour = currenthour+j
                    hr.append((hour+1)%24)
                    tr.append((currentdt + datetime.timedelta(hours=hour)).strftime('%Y/%m/%d %H:%M:%S'))
                    yr.append(v)
                    fr.append(1)
            i = i + 1
        hr.append(Time._converthour(t[i]))
        tr.append(DateTimeReader.todatetime(t[i]).strftime('%Y/%m/%d %H:%M:%S'))
        yr.append(y[i])
        fr.append(0)
        yr = Arr.nan_fill(yr,fill=filltype)
        return hr,tr,yr,fr

    #factory method to return time partitions
    #example:
    # without skip:
    # partition([1,2,0,1,2,0,1],["a","b","c","d","e","f","g"],3)
    # 1201201
    # abcdefg
    # [cab],0,2
    # [cdb],1,3
    # [cde],2,4
    # [fde],3,5
    # [fge],4,6
    # with skip:
    # partition([1,2,0,1,2,0,1],["a","b","c","d","e","f","g"],3,skip=[0,0,0,1,0,0,0])
    # 1201201
    # abcdefg
    # [cab],0,2
    # [fge],4,6
    def partition(h,y,n=24,skip=None,slice=False):
        r = []
        rng = len(h)-n+1
        skipindex = [False]*rng
        if skip != None:
            for i in range(rng):
                if 1 in skip[i:i+n]:
                    skipindex[i] = True
        if slice:
            for i in range(rng):
                if skipindex[i]:
                    continue
                if h[i] == 0:
                    r.append(TimePartition(y[i:i+n],i,i+n-1))
            return r
        else:
            for i in range(rng):
                if skipindex[i]:
                    continue
                h1 = h[i:i+n]
                y1 = y[i:i+n]
                y2 = Arr.remap(h1,y1,reverse=True)
                r.append(TimePartition(y2,i,i+n-1))
            return r

    #count number of hours needed to shift y to sync with x
    def hourshift(x,y):
        xhour = DateTimeReader.todatetime(x[0]).hour
        for i in range(24):
            yhour = DateTimeReader.todatetime(y[i]).hour
            if xhour == yhour:
                return i

    def tohourarray(x):
        return [DateTimeReader.todatetime(v).hour for v in x]

    def _delim(t):
        if "/" in t:
           delim = "/"
        elif "-" in t:
           delim = "-"
        else:
           raise Exception("datetime conversion error: invalid delimiter")
        return delim

    #convert to datetime object
    def todatetime(t):
        delim = Time._delim(t)
        try:
            t = datetime.datetime.strptime(t,"%Y" + delim + "%m" + delim + "%d %H:%M:%S")
        except ValueError:
            t = datetime.datetime.strptime(t,"%Y" + delim + "%m" + delim + "%d %H:%M")
        return t

    #return time as hour ordinal
    def hour_ord(t):
        if not isinstance(t[0], (datetime.datetime,)):
            t = [Time.todatetime(v) for v in t]
        return [v.toordinal()*24+v.hour for v in t]

    #convert hour ordinal to string
    def hour_ord_tostr(t,delim):
        return [datetime.datetime.fromordinal(int(v/24)).strftime("%Y" + delim + "%m" + delim + "%d") + " " + str(v%24) + ":00:00" for v in t]

    #fill missing rows in dataframe
    #df_filled: filled dataframe
    #idx_filled: filled indices
    def df_fill(df):
        df_filled = pd.DataFrame.copy(df)
        delim = Time._delim(df_filled["DateTime"][0])
        df_filled.index = Time.hour_ord(df_filled["DateTime"])
        shift = min(df_filled.index)
        new_index = np.arange(shift, max(df_filled.index)+1, 1)
        idx_filled = sorted(set(df_filled.index) ^ set(new_index))
        df_filled = df_filled.reindex(new_index)
        df_filled.ix[idx_filled,"DateTime"] = Time.hour_ord_tostr(idx_filled,delim)
        idx_filled = [v-shift for v in idx_filled]
        df_filled.reset_index(drop=True, inplace=True)
        return df_filled,idx_filled

class TimePartition():
    def __init__(self,data,index_start,index_end):
        self.data = data
        self.index_start = index_start
        self.index_end = index_end
        self.date_start = None
        self.date_end = None

    def __str__(self):
        r = str(self.data)
        r = r + "(" + str(self.index_start) + "," + str(self.index_end) + ")"
        if self.date_start != None and self.date_end != None:
            r = r + "(" + str(self.date_start) + "," + str(self.date_end) + ")"
        return r

    def getdata(self):
        return self.data

    def setdata(self, data):
        self.data = data

    def getindex(self):
        return self.index_start,self.index_end

    def setindex(self, index_start, index_end):
        self.index_start = index_start
        self.index_end = index_end

    def getdate(self):
        return self.date_start,self.date_end

    def setdate(self, dates):
        self.date_start = dates[self.index_start]
        self.date_end = dates[self.index_end]

class Trough():
    def __init__(self,x,i1,i2):
        self.i1 = i1
        self.i2 = i2
        self.length = i2-i1
        self.edge, self.abs_ratio, self.rel_ratio = Trough.fillratio(x,i1,i2)
        self.children = []

    def __key__(self):
        return str(self)

    def __hash__(self):
        return hash(self.__key__())

    def __eq__(self,other):
        return (self.i1 == other.i1) and (self.i2 == other.i2)

    def __lt__(self,other):
        return self.length < other.length

    def __str__(self):
        rng = str(self.i1) + "," + str(self.i2)
        chi = str(len(self.children))
        rat = "abs:" + str(self.abs_ratio) + "," + "rel:" + str(self.rel_ratio)
        return rng + ";" + chi + ";" + rat

    #return fill ratio
    def fillratio(x,i1,i2):
        if i1 == -1 or i2 == len(x):  #edge case
            if i1 == -1:
                i1 = 0
            if i2 == len(x):
                i2 = len(x)-1
            edge = max(x[i1],x[i2])
        else:
            edge = min(x[i1],x[i2])
        temp = [edge] + x[i1+1:i2] + [edge]
        abs_ratio = np.mean(temp)/temp[0]
        rel_ratio = (np.mean(temp)-min(temp))/(temp[0]-min(temp))
        return edge,abs_ratio,rel_ratio

    def copy(self,other):
        self.i1 = other.i1
        self.i2 = other.i2
        self.length = other.length
        self.edge = other.edge
        self.abs_ratio = other.abs_ratio
        self.rel_ratio = other.rel_ratio

    #root tree to list
    def tolist(self,a=[]):
        a.append(self)
        for c in self.children:
            c.tolist(a)
        return a

    #print root tree
    def print(self,pre=""):
        print(pre + str(self))
        for c in self.children:
            if len(self.children) > 1:
                c.print(pre+"|")
            else:
                c.print(pre+" ")

    #fit child to parent
    def fitchildtoparent(self,child):
        for c in self.children:
            if c.fitchildtoparent(child):
                return True  #child is a child of child
        if child.i1 >= self.i1 and child.i2 <= self.i2:  #try to adopt child
            self.children.append(child)
            return True
        return False

    def prune(self):
        if len(self.children) == 0:
            return self
        singlechildren = [self]
        nextchildren = self.children
        while len(nextchildren) == 1:
            singlechild = nextchildren[0]
            singlechildren.append(singlechild)
            nextchildren = singlechild.children
        singlechildren = sorted(singlechildren, key=lambda x: x.rel_ratio)
        self.copy(singlechildren[0])
        self.children = nextchildren
        for i in range(len(self.children)):
            self.children[i] = self.children[i].prune()
        return self

    def diff(self,x):
        if x[self.i1] > x[self.i2]:
            temp = [x[self.i2]] + x[self.i1+1:self.i2+1]
        else:
            temp = x[self.i1:self.i2] + [x[self.i1]]
        return temp[0]-np.mean(temp)

    #left search
    def left(x,minlen=0):
        a = []
        i = len(x)-1
        while i > -1:
            l = i - 1
            while l > -1:
                if x[l] >= x[i] or Proximate.float(x[l],x[i]):
                    break
                l = l - 1
            if l+1 < i and i-l > minlen:
                a.append(Trough(x,l,i))
            i = i - 1
        return a

    #right search
    def right(x,minlen=0):
        a = []
        i = 0
        while i < len(x):
            r = i + 1
            while r < len(x):
                if x[r] >= x[i] or Proximate.float(x[r],x[i]):
                    break
                r = r + 1
            if i+1 < r and r-i > minlen:
                a.append(Trough(x,i,r))
            i = i + 1
        return a

    def search(x,minlen=10,thr_rel_fill=0.5,thr_abs_fill=0.9,rootfilter=True):
        x = list(x)
        l = Trough.left(x,minlen)
        r = Trough.right(x,minlen)
        t = sorted(list(set(l+r)),reverse=True)  #remove duplicates and sort by longest interval
        #Trough.graph(x,t)
        for i in range(len(t)-1,-1,-1):
            if t[i].abs_ratio > thr_abs_fill:
                del t[i]
            elif t[i].rel_ratio > thr_rel_fill:
                del t[i]
        Trough.graph(x,t)
        if rootfilter:
            root = Trough(x,-1,len(x))
            root.edge = max(x)
            for child in t:
                root.fitchildtoparent(child)
            root = root.prune()
            t = root.tolist()
            #root.print()
            #Trough.graph(x,t)
        return t

    def graph(x,t):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines([i for i in range(len(x))], 0, x)
        y = 1
        for trough in t:
            ax.hlines(trough.edge,trough.i1,trough.i2,color="red")
            y = y - 0.1
        plt.show()

class Vec():
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Vec: " + "(" + str(self.x) + "," + str(self.y) + ")"

    def dot(self,v):
        return self.x*v.x+self.y*v.y

    def xsign(self):
        if self.x < 0:
            return -1
        else:
            return 1

    def ysign(self):
        if self.y < 0:
            return -1
        else:
            return 1

    def magnitude(self):
        return math.sqrt(self.x*self.x+self.y*self.y)

    def sub(self,v):
        return Vec(self.x-v.x,self.y-v.y)

    def mul(self,c):
        return Vec(self.x*c,self.y*c)

    def unit(self):
        m = self.magnitude()
        if m == 0:
            return Vec(0,0)
        return Vec(self.x/m,self.y/m)

    #self projection on v
    def projection(self,v):
        u = v.unit()
        return u.mul(self.dot(u))

    #self rejection on v
    def rejection(self,v):
        return self.sub(self.projection(v))

    def print(self):
        print("x:",self.x,",","y:",self.y)

    #return the projection distance
    #example:
    # x = [0,2,2]
    # y = [0,0,2]
    # pdist(x,y) = [0.0,1.4,2.8]
    def pdist(x,y):
        b = Vec(x[-1]-x[0],y[-1]-y[0])
        ds = [0]*len(x)
        for i in range(0,len(x)):
            a = Vec(x[i]-x[0],y[i]-y[0])
            d = a.projection(b)
            ds[i] = d.magnitude()
        return ds

    #return the rejection distance
    #example:
    # x = [0,2,2]
    # y = [0,0,2]
    # rdist(x,y) = [0.0,1.4,0.0]  #distance between point (2,0) and (1,1)
    def rdist(x,y):
        b = Vec(x[-1]-x[0],y[-1]-y[0])
        ds = [0]*len(x)
        for i in range(0,len(x)):
            a = Vec(x[i]-x[0],y[i]-y[0])
            d = a.rejection(b)
            ds[i] = d.magnitude()*(-1*d.ysign())  #positive if under line; negative if over/equal line
        return ds

if __name__ == '__main__':
    Spectrum.test()

