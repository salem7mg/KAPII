#multiprocessing pool wrapper for starting asynchronous processes

import signal
import multiprocessing as mp

import sys
import time
import pandas as pd

from aux_math import Arr

#initialize mutex lock
def initializer(l):
    #signal.signal(signal.SIGINT, signal.SIG_IGN)  #for ctrl-c
    global lock
    lock = l

class Pooler():
    #return shared dictionary
    def shared_dict():
        manager = mp.Manager()
        sd = manager.dict()
        sd["exit"] = False
        sd["counter"] = 0
        return sd

    #start processes
    #f_proc: process function
    #f_poll: polling function
    #initializer: pool initializer
    #arglists: argument lists (ex:[(p1_arg1,p1_arg2),(p2_arg1,p2_arg2)])
    #sd: shared dictionary
    #n: number of processes
    def start_ps(f_proc,f_poll,initializer,arglists,sd,n=None):
        if n == None:
            n = mp.cpu_count() - 1
        l = mp.Lock()
        pool = mp.Pool(processes=n, initializer=initializer, initargs=(l,))
        results = []
        for arglist in arglists:
            results.append(pool.apply_async(f_proc, arglist + (sd,)))  #returns immediately
        rtemp = [r for r in results]
        try:
            while True:
                time.sleep(1)
                for r in rtemp:
                    if r.ready():
                        rtemp.remove(r)
                if not rtemp:
                    break
                if sd["exit"]:
                    pool.close()
                    pool.join()
                    return None
                f_poll()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            exit()
        try:
            results = [v.get() for v in results]
        except Exception as e:  #exception raised in worker process
            pool.close()
            pool.join()
            e.args += ("exception raised in worker process",)
            raise
        pool.close()
        pool.join()
        return results

    #split dataframe into equal parts
    #df: dataframe
    #columns: selected columns (list)
    #n: number of parts
    def split_df(df,columns,n=None):
        if n == None:
            n = mp.cpu_count() - 1
        r = []
        col_first = [df.columns[0]]
        col_split = Arr.split(columns,n)
        for col_part in col_split:
            r.append(df[col_first + col_part])
        return r

def test(dfp,sd):
    for i in range(100):
        time.sleep(0.1)
        lock.acquire()
        sd["counter"] = sd["counter"] + 1
        lock.release()
    return dfp.multiply(100)

if __name__ == '__main__':
    mp.freeze_support()
    sd = Pooler.shared_dict()
    df = pd.DataFrame(columns=["id","a","b","c"], data=[[0,1,5,0],[1,2,6,0],[2,3,7,0],[3,4,8,0]])
    results = Pooler.start_ps(test,lambda:print(sd["counter"]),initializer,[(dfp,) for dfp in Pooler.split_df(df,["a","b","c"])],sd)
    result = pd.concat([v.iloc[:,1:] for v in results], axis=1, join='inner')
    print(result)

