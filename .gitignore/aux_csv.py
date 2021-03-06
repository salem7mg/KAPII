﻿"""Auxiliary csv functions.
"""

import time
import os
import shutil
import pandas as pd

class CSV():
    SEPARATOR = ","
    ENCODING = "SJIS"

    #read header
    def readheader(path):
        df = CSV.read(path, nrows=1)
        df = df.dropna(axis=0, how='all')
        return df.columns.tolist()

    #read single column
    def readcolumn(path,header):
        #squeeze: return a Series
        df = CSV.read(path, header=0, usecols=[header], squeeze=True)
        df = df.dropna(axis=0, how='all')
        return df.values.tolist()

    #write single column
    def writecolumn(path,header,values):
        df = CSV.read(path)
        df = df.dropna(axis=0, how='all')
        df[header] = pd.DataFrame(values, columns=[header])
        CSV.write(df,path)

    #insert empty rows to df
    #df: dataframe
    #i: index to insert to (ex: if i=0 then insert before 0)
    #n: number of rows to insert (n>=1)
    def df_insrow(df,i,n=1):
        df1 = df.iloc[0:i]
        df2 = pd.DataFrame([[]]*n)
        df2.index = [j for j in range(i,i+n)]
        df3 = df.iloc[i:len(df)]
        df3.index = df3.index + n
        return df1.append(df2).append(df3)

    #replace df1 values with df2 values
    def df_rplval(df1,df2):
        rc = df2.columns.tolist()
        df1.loc[df1.index.isin(df2.index), rc] = df2[rc]
        return df1

    #concat dfs by column
    def df_concat_col(dfs):
        return pd.concat(dfs, axis=1, join="inner")

    #return number of rows
    def rows(filepath):
        row_count = 0
        with open(filepath,"r") as file:
            row_count = sum(1 for row in file)
        return row_count

    def _get_unique_path(dir,file):
        f = file
        path = os.path.join(dir,f)
        i = 0
        while os.path.exists(path):
            f = str(i) + "_" + file
            path = os.path.join(dir,f)
            i = i + 1
        return f,path

    #pd.read_csv wrapper - read csv to dataframe
    def read(filepath,tempdir="C:/",tempfile="temp.csv",header=0,**kwargs):
        df = None
        cwd = os.getcwd()
        os.chdir(tempdir)
        if not os.path.isfile(filepath):
            filepath = cwd + "\\" + filepath
        if os.path.isfile(filepath):
            if os.path.getsize(filepath) > 0:
                cfile,cpath = CSV._get_unique_path(tempdir,tempfile)  #find unique path to copy to
                shutil.copyfile(filepath, cpath)                      #copy file
                csize = -1                                            #wait until copy finishes
                fsize = os.stat(filepath).st_size
                while (csize != fsize):
                    csize = os.stat(cpath).st_size
                    time.sleep(1)
                rfile,rpath = CSV._get_unique_path(tempdir,cfile)     #find unique path to rename to
                os.rename(cpath, rpath)                               #rename file (renames are atomic)
                df = pd.read_csv(rfile, sep=CSV.SEPARATOR, encoding=CSV.ENCODING, header=header, **kwargs)
                #df = df.dropna(axis=0, how='all')
                os.remove(rpath)                                      #remove temp file
        os.chdir(cwd)
        return df

    #df.to_csv wrapper - write dataframe to csv
    def write(df,outputpath,header=True,**kwargs):
        df.to_csv(outputpath, index=False, sep=CSV.SEPARATOR, encoding=CSV.ENCODING, header=header, **kwargs)

    #concat csv files by row
    #return number of rows
    def concat_row(filepaths,outputpath,tempdir="C:/",tempfile="temp.csv",header=None):
        dfs = []
        for i,filepath in enumerate(filepaths):
            dfs.append(CSV.read(filepath,tempdir,str(i) + "_" + tempfile,header))
        df = pd.concat(dfs, ignore_index=True)
        n = len(df)
        CSV.write(df,outputpath,header)
        return n

if __name__ == '__main__':
    print("read")
    print(CSV.read("df.csv"))
    print(CSV.readheader("df.csv"))
    print(CSV.readcolumn("df.csv","a"))
    print(CSV.readcolumn("df.csv","b"))
    print(CSV.readcolumn("df.csv","c"))
    print("df_insrow")
    df = CSV.read("df.csv")
    print(CSV.df_insrow(df,0))
    print(CSV.df_insrow(df,1))
    print(CSV.df_insrow(df,2))
    print("df_rplval")
    df1 = pd.DataFrame(columns=["a", "b", "c"], data=[[1,2,3],[4,5,6],[7,8,9]])
    df2 = pd.DataFrame(columns=["a", "b"], data=[[0,0],[1,1]])
    print(df1)
    CSV.df_rplval(df1,df2)
    print(df1)
    print("concat_row")
    filepaths = ["df1.csv","df2.csv"]
    outputpath = os.getcwd() + "\\df_out.csv"
    print(CSV.concat_row(filepaths,outputpath))
    print(CSV.read(outputpath))
    print("df_concat_col")
    print(CSV.df_concat_col([CSV.read("df1.csv"),CSV.read("df2.csv")]))
