"""Miscellaneous.
Listfile        #read/write list from/to file
DateTimeReader  #datetime reader
PriorityQueue   #priority queue
Cmd             #command line functions
MProc           #multiprocessing
Filemngr        #file manager
Debugger        #run debugger
"""

import datetime
import math
import os
import re
import subprocess
import sys
import time
import numpy as np
import heapq  #https://docs.python.org/3/library/heapq.html

#multiprocessing
import multiprocessing  #multiprocessing.Queue() for processes (piping)
from multiprocessing.sharedctypes import Value, Array

class Misc():
    #get words from string (ignore strings with numbers)
    #example:
    # getwords("123 a b c") = ['a', 'b', 'c']
    def getwords(line):
        r = []
        arr = line.split()
        for word in arr:
            c = 0
            for ch in word:
                if ch.isalpha():
                    c = c + 1
            if c >= len(word):
                r.append(word)
        return r

class Listfile():
    def write(fname,x):
        with open(fname,"w") as f:
            for v in x:
                f.write(str(v) + "\n")
    def read(fname):
        a = []
        with open(fname,"r") as f:
            lines = f.readlines()
            for line in lines:
                if line.endswith("\n"):
                    v = line[:-1]
                else:
                    v = line
                if v != "":
                    a.append(v)
        return a

class DateTimeReader():
    def todatetime(t):
        return t
        try:
            t = datetime.datetime.strptime(t,'%Y/%m/%d %H:%M:%S')
        except ValueError:
            t = datetime.datetime.strptime(t,'%Y/%m/%d %H:%M')
        return t

class PriorityQueue():
    def __init__(self):
        self.q = []

    def push(self,a):
        heapq.heappush(self.q,a)

    def repush(self,a):
        self.remove_value(a)
        self.push(a)

    def pop(self):
        return heapq.heappop(self.q)

    def remove_index(self,i):
        self.q[i] = self.q[-1]
        self.q.pop()
        if i < len(self.q):
            heapq._siftup(self.q, i)
            heapq._siftdown(self.q, 0, i)

    def remove_value(self,a):
        self.q.remove(a)
        heapq.heapify(self.q)

    def heapify(self):
        heapq.heapify(self.q)

    def empty(self):
        return len(self.q)==0

    def print(self):
        print(self.q)

class Cmd():
    #enter command in command line
    #use file if output is longer than 4096 characters
    # child process will hang(wait) if parent subprocess pipe buffer is full
    # buffer overflow > 4096 (65536/4096 = 16bits per char)
    # will not hang (4094 + newline + null = 4096): command = "python -c \"print('*' * 4094)\""
    # will hang (4095 + newline + null = 4097): command = "python -c \"print('*' * 4095)\""
    def enter(command, file=None):
        if file == None:
            process = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out = []
            err = []
            for line in process.stdout:
                out.append(line.decode("Shift-JIS").rstrip())
            for line in process.stderr:
                err.append(line.decode("Shift-JIS").rstrip())
            for line in out:
                print(line)
            for line in err:
                print("err:"+line)
            return out, err
        else:
            log_out = open("log_out.txt", "w")
            log_err = open("log_err.txt", "w")
            p = subprocess.Popen(command,shell=True,stdout=log_out,stderr=log_err)
            while p.poll() == None:
                time.sleep(1)
            log_out.close()
            log_err.close()
            out = ""
            with open("log_out.txt", 'r') as f:
                line = f.read()
                print(line)
                out = out + line + "\n"
            err = ""
            with open("log_err.txt", 'r') as f:
                line = f.read()
                print("err:"+line)
                err = err + line + "\n"
            return out, err

    def netusemap(ip,drive="Z:",username="",password=""):
        Cmd.enter("net use " + drive + " /d /y")
        time.sleep(2)
        if username != "" and password != "":
            user = "/user:" + username + " " + password
        else:
            user = ""
        out, err = Cmd.enter("net use " + drive + " \\\\" + ip + "\\share " + user + " /p:yes")
        return (len(err) == 0)

    def getfilepaths(path, regexstr="^[a-z]*\.txt$"):
        out, err = Cmd.enter("dir \"" + path + "\" /b /a-d | findstr " + regexstr)
        return out

    #/d[:mm-dd-yyyy]
    def copyfile(path1,path2,date=""):
        if date != "":
            date = ":" + date
        out, err = Cmd.enter("xcopy \"" + path1 + "\" \"" + path2 + "\" /d" + date + " /y /z")
        return (out[1][:1] == "1")

class MProc():
    """Multiprocessing class.
    Note:
    To add freeze support (windows executable), add freeze_support() right after main.
    if __name__ == '__main__':
        multiprocessing.freeze_support()
    """
    def __init__(self):
        self.lock = multiprocessing.Lock()
        self.ctx = multiprocessing.get_context('spawn')
        #multiprocessing.set_start_method('spawn')
        self.task_q = self.ctx.Queue()
        self.done_q = self.ctx.Queue()

    def addtask(self,task):
        self.task_q.put(task)

    def print_p(p):
        print(p)
        print("is_alive:",p.is_alive())
        print("daemon:",p.daemon)
        print("pid:",p.pid)
        print("exitcode:",p.exitcode)

    def test_spawnpipe(c,i):
        c.send("pipe"+str(i))
        c.close()

    def test_spawn(q,l,i):
        l.acquire()
        try:
            pid = str(os.getpid())
            print("process#"+str(i)+" pid#"+pid+" starting")
        finally:
            l.release()
        time.sleep(1)
        q.put("process#"+str(i)+" pid#"+pid+" finished")

    def test_pool(x):
        time.sleep(1)
        return x*x

    def test_poolasync():
        with multiprocessing.Pool(processes=4) as p:
            res = p.apply_async(time.sleep, (10,))
            try:
                print(res.get(timeout=1))
            except multiprocessing.TimeoutError:
                print("timeout")

    def test(self):
        print("-----testing spawn")
        ps = [self.spawn(MProc.test_spawn,i) for i in range(4)]
        print("starting process...")
        for x in ps:
            x.start()
        for x in ps:
            MProc.print_p(x)
        for x in ps:
            print(self.done_q.get())
        print("joining process...")
        for x in ps:
            x.join()
        print("-----testing spawnpipe")
        ps = [self.spawnpipe(MProc.test_spawnpipe,i) for i in range(4)]
        for x in ps:
            x["process"].start()
        for x in ps:
            print(x["parent_conn"].recv())
        print("joining process...")
        for x in ps:
            x["process"].join()
        print("-----testing pool")
        start = time.clock()
        print(self.pool(5,MProc.test_pool,[1,2,3,4,5]))
        print("pool time:",time.clock() - start)
        print("-----testing poolasync")
        MProc.test_poolasync()

    def pool(self, n, f_process, arglist):
        with multiprocessing.Pool(processes=n) as p:
            return p.map(f_process, arglist)

    #shared memory - pass by arg (ex: v = MProc.shared_value(1.0); v.value = 2.0)
    def shared_value(v):
        return Value("d",v)

    #shared memory - pass by arg (ex: a = MProc.shared_array([1,2,3]); a[0] = 4)
    def shared_array(a):
        return Array("i",a)

    def spawn(self, f_process, *args):
        arglist = (self.done_q, self.lock) + args
        p = self.ctx.Process(target=f_process, args=arglist)
        p.daemon = True  #exit when main process exits
        return p

    def spawnpipe(self, f_process, *args):
        parent_conn, child_conn = multiprocessing.Pipe()
        arglist = (child_conn,) + args
        return {"parent_conn":parent_conn,"child_conn":child_conn,"process":self.ctx.Process(target=f_process, args=arglist)}

class Filemngr():
    """General directory/file path functions.
    """
    #convert string to file string
    #naming conventions: https://msdn.microsoft.com/en-us/library/aa365247
    def tofilestr(str):
        filestr = str
        for symbol in ["<",">",":","\"","/","\\","|","?","*"]:
            filestr = filestr.replace(symbol,"")
        i = len(filestr)-1
        while i >= 0:
            if filestr[i] == "." or filestr[i] == " ":
                i = i - 1
            else:
                break
        return filestr[0:i+1]

    #make directory
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    #retrieve folder path from path
    #Filemngr.getfolder("C:\\1\\2\\3.4.5") = "C:\1\2"
    def getfolder(path):
        for c in ["\\","/"]:
            if c in path:
                s = path.split(c)
                if s[-1].count(".") > 0:
                    return c.join(s[:-1])
        return path

    #retrieve file name from path
    #Filemngr.getfile("C:\\1\\2\\3.4.5") = "3.4.5"
    def getfile(path):
        for c in ["\\","/"]:
            if c in path:
                return path.split(c)[-1]
        return path

    def opendir(path):
        if os.path.isdir(path):
            path = path.replace("/","\\")
            p = subprocess.Popen("explorer \"" + path + "\"")
            #(output, err) = p.communicate()
            #p_status = p.wait()
            #p.kill()
            #p.terminate()
        else:
            print("directory not found: " + path)

    def opennote(path):
        if os.path.isfile(path):
            path = path.replace("/","\\")
            p = subprocess.Popen("notepad \"" + path + "\"")
        else:
            print("file not found: " + path)

    #if subpath is found, use subpath as current dir; otherwise use curpath as current dir
    def getdir(subfoldername):
        curpath = os.getcwd()
        subpath = os.path.abspath(os.path.join(curpath, subfoldername))
        if os.path.exists(subpath):  #current directory is not subpath
            pdir = curpath
            cdir = subpath
        else:                        #current directory is subpath
            pdir = os.path.abspath(os.path.join(curpath, os.pardir))
            cdir = curpath
        return pdir.replace("\\","/"), cdir.replace("\\","/")

    #usage:
    #in program.spec file : in a = : datas = [('image.gif', '.'),...]
    #in py file : path = Filemngr.resource_path("image.gif")
    #to install : pyinstaller.exe program.spec
    def resource_path(filename):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, filename)  #sys._MEIPASS = temp path for pyinstaller
        return os.path.join(os.path.abspath("."), filename)

    #return a list of files
    def getfilelist(dir,filetype=".csv",filter="",fullpath=False):
        files = os.listdir(dir)
        r = []
        for file in files:
            if file[len(file)-len(filetype):len(file)] == filetype:
                if re.search("("+filter+")",file):
                    if fullpath:
                        r.append(os.path.join(dir,file))
                    else:
                        r.append(file)
        return r

class Debugger():
    def run():
        try:
            from IPython.core.debugger import Pdb
        except ImportError as e:
            print(e)
            from pdb import Pdb
            Pdb().set_trace()
            return
        Debugger.help()
        #
        #
        #
        #
        Pdb(context=9).set_trace()
        #
        #
        #
        #

    def help():
        print("debug commands:                   ")
        print(" show lines:       l              ")
        print(" step over:        n              ")
        print(" step in:          s              ")
        print(" step out:         r              ")
        print(" skip loop:        until          ")
        print(" continue:         c              ")
        print(" breakpoint:       b lineno       ")
        print(" clear breakpoint: clear no       ")
        print(" print:            p obj          ")
        print(" local vars:       locals()       ")
        print(" global vars:      globals()      ")
        print(" object type:      type(obj)      ")
        print(" object vars:      vars(obj)      ")
        print(" object methods:   dir(obj)       ")
        print(" help:             Debugger.help()")

    def reload():
        import importlib
        import aux_misc
        importlib.reload(aux_misc)
        from aux_misc import Debugger
        print("if using python interpreter type the following: \"from aux_misc import Debugger\"")

    def print(obj):
        print("-----type-----")
        print(type(obj))
        print("-----vars-----")
        try:
            v = vars(obj)
            for k in v.keys():
                print(k,"=",v[k])
        except TypeError as e:
            print(e)
        print("-----methods-----")
        for m in [method_name for method_name in dir(obj) if callable(getattr(obj, method_name))]:
            f = getattr(obj, m)
            if hasattr(f, '__code__'):
                print(m,f.__code__.co_varnames)
            else:
                print(m)
