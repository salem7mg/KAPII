"""Auxiliary GUI components.
from aux_tkinter import ThreadingException  #threading exception
from aux_tkinter import ThreadedGUI         #create a threaded gui application
from aux_tkinter import NewThread           #create a new thread
from aux_tkinter import StyleManager        #manage ttk styles
from aux_tkinter import HyperlinkManager    #manage hyperlinks in text widgets
from aux_tkinter import ScrollText          #text widget with scrollbar
from aux_tkinter import Dialogbox           #customized dialogbox
from aux_tkinter import PixelButton         #button with customizable size
from aux_tkinter import NewWindowButton     #button that creates a new window
from aux_tkinter import SinkButton          #button that sinks
from aux_tkinter import ToggleButton        #button that toggles
from aux_tkinter import ColorCanvas         #canvas with colors
from aux_tkinter import PlotCanvas          #canvas with plots
from aux_tkinter import DirEntry            #entry that accepts directories
from aux_tkinter import NumberEntry         #entry that accepts numbers
from aux_tkinter import Menubar             #menubar
from aux_tkinter import TabView             #tab
from aux_tkinter import TreeView            #tree
from aux_tkinter import DatePane            #date entry
from aux_tkinter import RadioSet            #set of radio buttons
from aux_tkinter import ImageLabel          #image
from aux_tkinter import ImagePane           #pane with image
from aux_tkinter import LabelPane           #pane with label
from aux_tkinter import Meter               #progress bar
from aux_tkinter import Slider              #slider
from aux_tkinter import DebugPane           #debug
"""

from aux_misc import Filemngr

import datetime
import importlib        #dynamic import
import os
import sys
import traceback        #exception traceback
import time
from numpy import arange, sin, pi

#threading
import _thread
import threading
import queue            #queue.Queue() for threads (mutex locking)

#tkinter
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.tix as tix
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox

#matplotlib
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class ThreadingException(Exception):
    """
    try:
        raise ThreadingException("test")
    except ThreadingException as e:
        print(e)
    """
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

    def handler(exception, value, tb):
        ThreadingException.msgbox()

    #return exc_info as string
    def exc_info():
        exception, value, tb = sys.exc_info()
        return ''.join(traceback.format_exception(exception, value, tb, limit=None))

    def msgbox(t_name=""):
        #filename = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        #lineno = str(tb.tb_lineno)
        #traceback.print_tb(tb)
        header = ""
        if t_name != "":
            header = "Thread name: " + t_name + "\n"
        title = "Exception"
        tk.messagebox.showerror(title,header + ThreadingException.exc_info())

#tkinter is not thread safe (always use main thread for tkinter components and branch out threads for processing)
#to catch exceptions not in queue (overrides master.report_callback_exception = ThreadingException.handler):
#gui.reset()
#try:
#    pass
#except Exception as e:
#    gui.reset()
#    ThreadingException.msgbox()
class ThreadedGUI():
    def __init__(self, master, w=500, h=500, sequential=True, f_stop=lambda:print("f_stop")):
        master.report_callback_exception = ThreadingException.handler
        self.master = master
        self.sequential = sequential
        self.f_stop = f_stop
        #set window size
        #master.resizable(width=False, height=False)
        #x = (master.winfo_screenwidth()-w)/2
        #y = (master.winfo_screenheight()-h)/2
        #master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.task_q = queue.Queue()  #task queue
        self.done_q = queue.Queue()  #completed tasks
        self.stop = False
        self.currentjob = None
        self.quitprogram = False
        self.error = None
        self.taskcount = 0           #number of spawned tasks
        self.donecount = 0           #number of completed tasks
        master.protocol("WM_DELETE_WINDOW", self.quit)
        master.after(100, self.process_queue)

    def reset(self):
        self.task_q.queue.clear()
        self.done_q.queue.clear()
        self.stop = False
        self.currentjob = None
        self.quitprogram = False
        self.error = None
        self.taskcount = 0
        self.donecount = 0

    def quit(self):
        self.quitprogram = True

    def thread_start(self, f_call=lambda:print("f_call"), f_main=lambda:print("f_main"), t_name=""):
        if self.error == None:
            self.taskcount = self.taskcount + 1
            job = NewThread(self.done_q, f_call, f_main, t_name)
            if self.sequential:
                self.task_q.put(job)
            else:
                job.start()
            return job
        else:
            raise self.error

    def thread_stop(self):  #for sequential threading only
        self.stop = True

    def process_queue(self):  #main polling
        try:
            if self.quitprogram:
                self.master.quit()
                return
            if self.sequential:
                if self.currentjob == None:
                    if self.stop == True:
                        print("stopping...")
                        self.task_q.queue.clear()
                        self.done_q.queue.clear()
                        self.stop = False
                        self.f_stop()
                        print("stopped")
                    elif self.error == None:
                        self.currentjob = self.task_q.get(0)
                        self.currentjob.start()
                elif (not self.currentjob.isAlive()) and self.currentjob.finished:  #check if thread is running/finished
                    self.donecount = self.donecount + 1
                    msg = self.done_q.get(0)                                        #blocking queue (get first message)
                    if type(msg) is ThreadingException:
                        self.error = msg                                            #propagate exception to exception handler (ThreadingException.handler)
                        self.stop = True
                    elif hasattr(msg, '__call__'):
                        f_main = msg
                        try:
                            f_main()
                        except Exception as e:
                            self.error = ThreadingException(e)
                            self.stop = True
                            ThreadingException.msgbox(self.currentjob.t_name)
                    self.currentjob = None
                self.master.after(100, self.process_queue)                          #wait 100ms (idle)
                #print("polling...")
            else:
                msg = self.done_q.get(0)                                            #blocking queue (get first message)
                f_main = msg
                f_main()
                #f_main(msg)                                                        #process message
                self.master.after(100, self.process_queue)                          #wait 100ms (idle)
        except queue.Empty:
            self.master.after(100, self.process_queue)                              #wait 100ms (idle)
            #print("idle...")

    def enable(self, level=1):
        self.chgstate("enabled", self.master, level)

    def disable(self, level=1):
        self.chgstate("disabled", self.master, level)

    def chgstate(self, state, master=None, level=1):
        if level == 0:
            return
        if master == None:
            master = self.master
        for w in master.winfo_children():
            self.chgstate(state, w, level-1)
            try:
                w.configure(state=state)
            except:
                pass

    def test():
        def t1():  #Exception in child thread (queue)
            gui.reset()
            try:
                gui.thread_start(lambda:print(1),lambda:print(2))
                gui.thread_start(lambda:1/0,lambda:print(3))
                gui.thread_start(lambda:print(4),lambda:print(5))
            except:
                print(12345)  #should not print (first time)
        def t2():  #Exception in parent thread (queue)
            gui.reset()
            try:
                gui.thread_start(lambda:print(1),lambda:print(2))
                gui.thread_start(lambda:print(3),lambda:1/0)
                gui.thread_start(lambda:print(4),lambda:print(5))
            except:
                print(12345)  #should not print (first time)
        def t3():  #Exception in parent thread
            gui.reset()
            try:
                gui.thread_start(lambda:print(1),lambda:print(2))
                print(3)
                1/0
                gui.thread_start(lambda:print(4),lambda:print(5))
            except Exception as e:
                print(12345)  #should print (overrides master.report_callback_exception = ThreadingException.handler)
                gui.reset()
                ThreadingException.msgbox()
        root = tix.Tk()
        root.title("test")
        gui = ThreadedGUI(root)
        b1 = ttk.Button(root, text="Exception in child thread (queue)", command = t1)
        b2 = ttk.Button(root, text="Exception in parent thread (queue)", command = t2)
        b3 = ttk.Button(root, text="Exception in parent thread", command = t3)
        b4 = ttk.Button(root, text="Reset", command = gui.reset)
        b1.pack()
        b2.pack()
        b3.pack()
        b4.pack()
        root.mainloop()

class NewThread(threading.Thread):
    id = 0
    def __init__(self, queue, f_call, f_main, t_name):
        NewThread.id = NewThread.id + 1
        self.id = NewThread.id
        self.f_call = f_call
        self.f_main = f_main
        self.t_name = t_name
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.daemon = True     #exit thread when main thread exits
        self.queue = queue
        print("Task created " + str(self.id))
        self.finished = False
    def run(self):
        #process
        try:
            #print("Task running " + str(self.id))
            r = self.f_call()
        except Exception as e:
            self.queue.put(ThreadingException(e))  #propagate exception to main thread (ThreadedGUI process_queue())
            self.finished = True
            ThreadingException.msgbox(self.t_name)
        else:
            self.queue.put(self.f_main)
            #self.queue.put(r)
            #self.queue.put("Task finished " + str(self.id))
            self.finished = True
    def stop(self):
        self._stop_event.set()
    def stopped(self):
        return self._stop_event.is_set()
    def test():
        def aplus():
            global a
            a = a + 1
        global a
        a = 0
        done_q = queue.Queue()
        print(done_q.empty())
        job1 = NewThread(done_q,aplus,aplus)
        job2 = NewThread(done_q,aplus,aplus)
        print("a=",a)  #a=0
        print(done_q.queue)
        job1.start()
        print("a=",a)  #a=1
        print(done_q.queue)
        job2.start()
        print("a=",a)  #a=2
        print(done_q.queue)

class StyleManager():
    """Manage ttk styles.
    """
    class __StyleManager:
        def __init__(self):
            self.styles = set()
            pass
        def __str__(self):
            return repr(self)
    types = ["TButton", "TCheckbutton", "TCombobox", "TEntry", "TFrame", "TLabel", "TLabelFrame", "TMenubutton", "TNotebook", "TPanedwindow", "TRadiobutton", "Treeview", "Horizontal.TProgressbar", "Vertical.TProgressbar", "Horizontal.TScale", "Vertical.TScale", "Horizontal.TScrollbar", "Vertical.TScrollbar"]
    instance = None
    def __init__(self):
        if not StyleManager.instance:
            StyleManager.instance = StyleManager.__StyleManager()
    def __getattr__(self, name):
        return getattr(self.instance, name)
    def __setattr__(self, name):
        return setattr(self.instance, name)
    def getstyle(self, type, fg, bg, **kwargs):
        if type not in StyleManager.types:
            print("warning: type " + type + " not in StyleManager.types")
        style = "style" + fg + bg + "." + type
        if style not in self.instance.styles:  #create style
            s = ttk.Style()
            s.configure(style, foreground=fg, background=bg, **kwargs)
            self.instance.styles.add(style)
        return style
    def printstyles(self):
        print(self.instance.styles)

#from http://effbot.org/zone/tkinter-text-hyperlink.htm
class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground="blue", underline=1)
        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)
        self.reset()

    def reset(self):
        self.links = {}

    def add(self, action):
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event):
        self.text.config(cursor="hand2")

    def _leave(self, event):
        self.text.config(cursor="")

    def _click(self, event):
        for tag in self.text.tag_names(tk.CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return

class ScrollText(ttk.PanedWindow):
    def __init__(self, master, w, h, fg, bg, lpad=5, rpad=5):
        super().__init__(master)
        self.configure(style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.pack(padx=(lpad,rpad))
        self.text = tk.Text(self, width=w, height=h, borderwidth=3, relief="sunken", foreground=fg, background=bg, font=("Gothic",12))
        self.text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        bar = tk.Scrollbar(self, command=self.text.yview)
        bar.grid(row=0, column=1, sticky='nsew')
        self.text.config(yscrollcommand=bar.set)
        self.hlm = HyperlinkManager(self.text)
        self.minwidth = w

    def inserttext(self, text, f_call=None):
        textlen = ScrollText.counthalfwidth(text)
        if textlen > self.minwidth:
            self.minwidth = textlen + 2  #add 2 spaces after text
            self.text.config(width=self.minwidth)
        if f_call == None:
            self.text.insert(tk.INSERT, text)
        else:
            self.text.insert(tk.INSERT, text, self.hlm.add(f_call))

    def insertend(self):
        self.text.config(state=tk.DISABLED)

    def counthalfwidth(str):
        count = 0
        for c in str:
            if ord(c) < 128:
                count += 1
            else:
                count += 2
        return count

class Dialogbox():
    CNTR_WIN = 0x0001
    CNTR_SCREEN = 0x0002

    def __init__(self, master, title, text, fg="#57A4C8", bg="white", image=None):
        self.master = master
        self.fg = fg
        self.bg = bg
        self.root = tix.Toplevel(master)
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(width=False, height=False)
        self.frame = tk.Frame(self.root)
        self.frame.pack(expand=tk.YES, fill=tk.BOTH)
        self.toppane = ttk.PanedWindow(self.frame,style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.toppane.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        if image == None:
            lpane = self.toppane
        else:
            self.ipane = ImagePane(self.toppane, image)
            self.ipane.pack(anchor=tk.NW)
            lpane = self.ipane.rpane
        self.label = ttk.Label(lpane, text=text, style=StyleManager().getstyle("TLabel",fg,bg))
        self.label.pack(padx=5,pady=5)
        self.botpane = ttk.PanedWindow(self.frame,style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.botpane.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
        self.st = None            #scroll text
        self.buttons = []         #buttons
        self.root.lift()
        self.root.grab_set()      #focus dialog

    def centerwindow(self, flag=None, window=None):
        #set window if not set
        if window == None:
            window = self.master
        #set flag if not set
        if flag not in [Dialogbox.CNTR_WIN, Dialogbox.CNTR_SCREEN]:
            flag = Dialogbox.CNTR_WIN
        #screen size
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        #window
        wx = window.winfo_x()
        wy = window.winfo_y()
        ww = window.winfo_width()
        wh = window.winfo_height()
        #dialog window
        self.root.update()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if flag == Dialogbox.CNTR_WIN:
            #center to window
            x = int((ww-w)/2+wx)
            y = int((wh-h)/2+wy)
        elif flag == Dialogbox.CNTR_SCREEN:
            #center to screen
            x = int((sw-w)/2)
            y = int((sh-h)/2)
        else:
            #move to top left
            x = 0
            y = 0
        #check screen limits
        if x < 0:
            x = 0
        elif x > sw - w:
            x = sw - w
        if y < 0:
            y = 0
        elif y > sh - h:
            y = sh - h
        self.root.geometry(str(w) + "x" + str(h) + "+" + str(x) + "+" + str(y))

    def inserttext(self, text, f_call=None):
        if self.st == None:
            self.label.update()
            pixelwidth = self.label.winfo_width()
            charwidth = 8
            w = int(pixelwidth/charwidth)
            #charheight = 20
            h = 5
            self.st = ScrollText(self.botpane,w,h,self.fg,self.bg,100)
            self.st.pack(side=tk.RIGHT)
        self.st.inserttext(text, f_call)

    def insertbutton(self, text, command):
        if len(self.buttons) == 0:
            self.buttongrid = ttk.PanedWindow(self.botpane,style=StyleManager().getstyle("TPanedwindow",self.fg,self.bg))
            self.buttongrid.pack()
        b = tk.Button(self.buttongrid, text=text, command=command)
        b.grid(row=0,column=len(self.buttons),padx=5,pady=5)
        self.buttons.append(b)

    def insertend(self):
        self.st.insertend()

    def on_close(self):
        self.root.grab_release()  #unfocus dialog
        self.root.destroy()

class PixelButton(tk.Frame):
    """A button with a size that can be set by pixels.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    width : int
        Width of button by pixels.
    height : int
        Height of button by pixels.
    """
    def __init__(self, master, width=0, height=0, **kwargs):
        self.width = width
        self.height = height
        tk.Frame.__init__(self, master, width=self.width, height=self.height)
        self.button = tk.Button(self, **kwargs)
        self.button.pack(expand=tk.YES, fill=tk.BOTH)

    def pack(self, *args, **kwargs):
        tk.Frame.pack(self, *args, **kwargs)
        self.pack_propagate(False)

    def grid(self, *args, **kwargs):
        tk.Frame.grid(self, *args, **kwargs)
        self.grid_propagate(False)

    def enable(self):
        self.button["state"] = "normal"

    def disable(self):
        self.button["state"] = "disabled"

class NewWindowButton(PixelButton):
    """A button that creates a new window.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    text : str
        Button text.
    f_on_create : function
        Function to call on new window create.
    b_create : function
        Function to call before create window.
    b_close : function
        Function to call before close window.
    title : str
        Title of new window.
    w : int
        Width of new window.
    h : int
        Height of new window.
    Attributes
    ----------
    master : Tk object
        Tk object to use as master.
    title : str
        Title of new window.
    w : int
        Width of new window.
    h : int
        Height of new window.
    root : Tk object
        New window.
    """
    f_on_create_default = lambda self:self.on_create()
    def __init__(self, master, text, f_on_create=f_on_create_default, b_create=None, b_close=None, title="New window", w=500, h=500, fg="black", bg="white", b_width=50, b_height=20):
        super().__init__(master, fg=fg, bg=bg, width=b_width, height=b_height)
        self.master = master  #parent window
        self.f_on_create = f_on_create
        self.b_create = b_create
        self.b_close = b_close
        self.title=title
        self.w = w
        self.h = h
        self.pack(ipadx=0, ipady=0)
        self.button.configure(text=text,command=lambda: self.create())
        self.open = False

    def create(self):
        """Create new window.
        """
        if self.b_create != None:
            self.b_create()
        #disable button
        self.button.config(state="disabled")
        self.setopen(True)
        #create new window
        self.root = tix.Toplevel(self.master)  #child window
        self.root.title(self.title)
        self.root.geometry(str(self.w) + "x" + str(self.h))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)     #call on_close when closing window
        #choose function to call
        import operator
        cc = operator.attrgetter('co_code', 'co_consts')
        if cc((self.f_on_create).__code__) == cc((self.f_on_create_default).__code__):
            print("loading default window")
            self.f_on_create()                                    #call on_create
        else:
            print("loading user window")
            self.f_on_create()                                    #call user function

    def close(self):
        """Close window.
        """
        if self.b_close != None:
            self.b_close()
        #enable button
        self.button.config(state="normal")
        self.setopen(False)
        self.root.destroy()

    def on_close(self):
        """Command on window close.
        """
        d = Dialogbox(self.root,"Quit","\nAre you sure you want to exit?\n",fg="black",bg="white")
        d.insertbutton("Yes", lambda: self.close() or d.on_close())
        d.insertbutton("No", lambda: d.on_close())
        d.centerwindow(Dialogbox.CNTR_WIN, self.root)

    def on_create(self):
        """Command on window create.
        """
        mbar = Menubar(self.root,self.on_close)

    def on_reset(self):
        """Command on window reset.
        """
        for child in self.root.winfo_children():
            if not isinstance(child,Menubar):
                child.destroy()

    def setopen(self,flag):
        """Set window open flag to True or False.
        """
        self.open = flag

    def isopen(self):
        """Return window open flag.
        """
        return self.open

class SinkButton(tk.Button):
    def __init__(self, master, text, cmd=lambda args=None: SinkButton.action(args)):
        super().__init__(master)
        self.configure(text=text, command=cmd)
        self.bind('<KeyPress>', self.onKeyPress)
    def onKeyPress(self, event):
        self.config(relief=tk.SUNKEN)
    def action(args):
        print("button pressed...")

class ToggleButton(tk.Button):
    def __init__(self, master, text1="cmd1", text2="cmd2", cmd1=lambda:print("cmd1") or time.sleep(1), cmd2=lambda:print("cmd2") or time.sleep(1), fg1="white", bg1="#EECD59", fg2="#57A4C8", bg2="white"):
        super().__init__(master)
        self.text1 = text1
        self.text2 = text2
        self.cmd1 = cmd1
        self.cmd2 = cmd2
        self.fg1 = fg1
        self.bg1 = bg1
        self.fg2 = fg2
        self.bg2 = bg2
        self.switch = True
        self.configure(text=self.text1, command=lambda:self.runcmd1(), foreground=self.fg1, background=self.bg1, relief='groove')
    def enable(self):
        self.configure(state=tk.NORMAL, foreground=self.fg1, background=self.bg1)
    def disable(self):
        self.configure(state=tk.DISABLED, foreground=self.fg2, background=self.bg2)
    def toggle(self, switch=None):
        if switch == None:
           switch = self.switch
        if switch:
            self.configure(text=self.text2, command=lambda:self.runcmd2(), foreground=self.fg1, background=self.bg1)
        else:
            self.configure(text=self.text1, command=lambda:self.runcmd1(), foreground=self.fg1, background=self.bg1)
        self.switch = not self.switch
    def runcmd1(self):
        self.disable()
        self.cmd1()
        self.toggle()
        self.enable()
    def runcmd2(self):
        self.disable()
        self.cmd2()
        self.toggle()
        self.enable()

class ColorCanvas(tk.Canvas):
    def __init__(self, master, w=255, h=255):
        super().__init__(master)
        self.width = w
        self.height = h
        self.configure(width=w, height=h, bg="#000000", highlightthickness=0, relief='ridge')
        self.pack()
        self.img = tk.PhotoImage(width=w, height=h)
        self.create_image((w/2, h/2), image=self.img, state="normal")
        #self.after(500, self.color)
    #https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
    #hue [0, 360]
    #saturation [0,1]
    #value [0,1]
    def hsvtorgb(h,s,v):
        h = h % 360
        chroma = s * v
        h1 = h / 60
        x = chroma * (1-abs((h1%2)-1))
        if (h1 >= 0 and h1 <= 1):
            r1,g1,b1 = chroma, x, 0
        elif (h1 >= 1 and h1 <= 2):
            r1,g1,b1 = x, chroma, 0;
        elif (h1 >= 2 and h1 <= 3):
            r1,g1,b1 = 0, chroma, x;
        elif (h1 >= 3 and h1 <= 4):
            r1,g1,b1 = 0, x, chroma;
        elif (h1 >= 4 and h1 <= 5):
            r1,g1,b1 = x, 0, chroma;
        elif (h1 >= 5 and h1 <= 6):
            r1,g1,b1 = chroma, 0, x;    
        m = v - chroma
        r,g,b = r1+m, g1+m, b1+m
        return ColorCanvas.itoh(int(255*r))+ColorCanvas.itoh(int(255*g))+ColorCanvas.itoh(int(255*b))
    #int[0, 255] to hex[00, ff]
    def itoh(i):
        return '{:02x}'.format(i)
    def color(self):
        #0xff00 = 65280
        #0xffffff = 16777215
        r = self.width*self.height
        for x in range(self.width):
            for y in range(self.height):
                self.img.put("#"+ColorCanvas.hsvtorgb(x*sin(y/10),(x*self.height+y)/r,1), (x,y))
                self.update()

class PlotCanvas():
    def __init__(self, master, s="sin(2*pi*t)", t=arange(0.0, 3.0, 0.01)):
        master = tix.Toplevel(master)
        f = Figure(figsize=(5, 4), dpi=100)
        a = f.add_subplot(111)
        s = eval(s)
        a.plot(t, s)
        a.set_title('Tk embedding')
        a.set_xlabel('X axis label')
        a.set_ylabel('Y axis label')
        self.canvas = FigureCanvasTkAgg(f, master)
        self.canvas.show()
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        self.toolbar.update()
        #self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def on_key_event(self, event):
        key_press_handler(event, self.canvas, self.toolbar)

    def kill(self):
        root.quit()
        root.destroy()

class DirEntry(ttk.PanedWindow):
    """A textbox with button that allows directory entry.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    title : str
        Title of file dialog window.
    fg : str
        Foreground color.
    bg : str
        Background color.
    wt : int
        Width of textbox.
    wb : int
        Width of button.
    """
    def __init__(self, master, title, fg, bg, wt=20, wb=10):
        super().__init__(master)
        self.title = title
        self.textbox = ttk.Entry(self, width=wt)
        self.textbox.insert(tk.END, "")
        self.textbox.pack(side=tk.LEFT, padx=5, expand=1, fill=tk.X)
        self.button = tk.Button(self, fg=fg, bg=bg, relief='groove', text='Open', command=self.open, width=wb)
        self.button.pack(side=tk.RIGHT, padx=5)
        self.pack(expand=1, fill=tk.X)

    def open(self):
        dir = tk.filedialog.askdirectory(title=self.title)
        if dir != "":
            self.textbox.delete(0, tk.END)
            self.textbox.insert(0, dir)

class NumberEntry(ttk.Entry):
    """A textbox that allows number entry.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    width : int
        Width of textbox.
    defaulttext : str
        Initial string in textbox.
    nchar : int
        Maximum number of characters.
    """
    allowed_keys = ("BackSpace","Delete","Left","Right","KF_Left","KP_Right")
    def __init__(self, master, width, defaulttext, nchar):
        super().__init__(master)
        self.width = width
        self.defaulttext = defaulttext
        self.nchar = nchar
        self.v = tk.StringVar()
        self.configure(state=tk.NORMAL, width=width, textvariable=self.v)
        self.insert(tk.END, defaulttext)
        self.pack(padx=2, side=tk.LEFT)
        self.bind("<FocusIn>", self.callback_focusin)
        self.bind("<FocusOut>", self.callback_focusout)
        self.bind("<KeyPress>", self.callback_keypress)
        self.bind("<Control-x>", self.callback_cut)
        self.bind("<Control-c>", self.callback_copy)
        self.bind("<Control-v>", self.callback_paste)
        self.bind("<Tab>", self.callback_tab)
        self.bind("<Shift-Tab>", self.callback_backtab)
    def callback_focusin(self,event):
        if self.get() == self.defaulttext:
            self.delete(0, tk.END)
    def callback_focusout(self,event):
        if self.get() == "":
            self.insert(0, self.defaulttext)
    def callback_keypress(self,event):
        if event.keysym in self.allowed_keys:
            return
        if event.keysym not in "0123456789":
            return "break"
        try:    
            if len(self.selection_get()) >= 1:
                return
        except:
            pass
        if len(self.get()) >= self.nchar:
            return "break"
    def callback_cut(self,event):
        return "break"
    def callback_copy(self,event):
        return "break"
    def callback_paste(self,event):
        return "break"
    def callback_tab(self,event):
        event.widget.tk_focusNext().focus()
        return "break"
    def callback_backtab(self,event):
        event.widget.tk_focusPrev().focus()
        return "break"

class Menubar(tix.Menu):
    """Window menubar.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    f_exit : function
        Function used to exit window.
    Attributes
    ----------
    master : Tk object
        Tk object to use as master.
    filemenu : Tix object
        File menu tix object.
    """
    ERR_DONOTHING = -1
    def __init__(self, master, f_exit, filepath="", f_load=None, f_save=None, dir_load=None, dir_save=None, arg=None):
        super().__init__(master)
        self.master = master
        self.f_exit = f_exit
        self.f_load = f_load
        self.f_save = f_save
        if dir_load != None:
           self.dir_load = dir_load
        else:
           self.dir_load = "C:/"
        if dir_save != None:
           self.dir_save = dir_save
        else:
           self.dir_save = "C:/"
        self.arg = arg
        self.filemenu = tix.Menu(self, tearoff=0)
        self.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="Load", command=self.loadcommand, accelerator="Ctrl+O")
        self.filemenu.add_command(label="Save", command=self.savecommand, state="disabled", accelerator="Ctrl+S")
        self.filemenu.add_command(label="Save As", command=self.saveascommand, state="disabled")
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=f_exit)
        if filepath.strip() == "":
            filepath = " "
        self.filepath = filepath
        self.filename = Filemngr.getfile(filepath)
        self.add_separator()
        self.separatorindex = self.index("File")+1
        self.add_command(label=self.filename, state="disabled")
        self.master.config(menu=self)
        self.bind_all("<Control-o>", self.loadevent)
        self.bind_all("<Control-s>", self.saveevent)
        self.newmenus = {}
        self.loadfile(filepath)
        #self.after(1000,lambda:self.loadfile(filepath))
        #menu = self.newmenu("test")
        #menu.add_command(label="test", command=lambda:print("test"))

    def entryexists(self, label):
        try:
            index = self.index(label)
        except tk.TclError as e:
            #print(e)
            return False
        else:
            return True

    def getmenu(self, label):
        return self.newmenus[label]

    def newmenu(self, label):
        if label not in self.newmenus:
            if not self.entryexists(label):
                menu = tix.Menu(self, tearoff=0)
                self.insert_cascade(self.separatorindex, label=label, menu=menu)
                self.separatorindex += 1
                self.newmenus[label] = menu
                return menu

    def delmenu(self, label):
        if label in self.newmenus:
            if self.entryexists(label):
                index = self.index(label)
                self.delete(index)
                if index < self.separatorindex:
                    self.separatorindex -= 1
                del self.newmenus[label]

    def loadevent(self, event):
        if self.filemenu.entrycget("Load","state")=="normal":
            self.loadcommand()

    def saveevent(self, event):
        if self.filemenu.entrycget("Save","state")=="normal":
            self.savecommand()

    def loadcommand(self):
        self.master.focus_force()
        filepath = tk.filedialog.askopenfilename(parent=self.master, initialdir=self.dir_load, filetypes = (("INI files", "*.ini"),("All files", "*.*")))
        self.loadfile(filepath)

    def savecommand(self):
        self.master.focus_force()
        self.savefile(self.filepath)

    def saveascommand(self):
        self.master.focus_force()
        filepath = tk.filedialog.asksaveasfilename(parent=self.master, initialdir=self.dir_save, filetypes = (("INI files", "*.ini"),("All files", "*.*")), defaultextension=".ini")
        self.savefile(filepath)

    def loadfile(self,filepath):
        if filepath.strip() != "" and self.f_load != None:
            try:
                #self.on_reset()
                children = self.getchildren()
                ret_val = self.f_load(self,filepath)                       #pass self and filepath (self=menu, self.master=current window, self.arg=arg)
                if ret_val == Menubar.ERR_DONOTHING:
                    return True
                temp = Filemngr.getfile(filepath)
                self.entryconfig(self.filename, label=temp)
                self.filepath = filepath
                self.filename = temp
                self.filemenu.entryconfig("Save", state="normal")
                self.filemenu.entryconfig("Save As", state="normal")
                for child in children:
                    child.destroy()
                return True
            except Exception as e:
                print(str(e))
                tk.messagebox.showerror("Open Source File", "Failed to read file \n'%s'"%filepath)
                #self.filemenu.entryconfig("Save", state="disabled")
                #self.filemenu.entryconfig("Save As", state="disabled")
                return False
        else:
            return False

    def savefile(self,filepath):
        if filepath.strip() != "" and self.f_save != None:
            try:
                self.f_save(self, filepath)                      #pass self and filepath (self=menu, self.master=current window, self.arg=arg)
                temp = Filemngr.getfile(filepath)
                self.entryconfig(self.filename, label=temp)
                self.filepath = filepath
                self.filename = temp
                return True
            except Exception as e:
                print(str(e))
                tk.messagebox.showerror("Open Source File", "Failed to read file \n'%s'"%filepath)
                return False
        else:
            return False

    def getchildren(self):
        children = []
        for child in self.master.winfo_children():
            if not isinstance(child,Menubar):
                children.append(child)
        return children

    def on_reset(self):
        for child in self.master.winfo_children():
            if not isinstance(child,Menubar):
                child.destroy()


class TabView(ttk.Notebook):
    """Tabbed frames.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    tabnames : list
        List of tab names.
    tabwidth : int
        Width of tab.
    Attributes
    ----------
    pages : list
        List of ttk frames.
    """
    def __init__(self, master, fg="#57A4C8", bg="white", tabnames=None, tabwidth=0):
        super().__init__(master)
        s = ttk.Style()
        s.configure("TNotebook", foreground=fg, background=bg, font=("Gothic",12,"bold"))
        s.configure("TNotebook.Tab", foreground=fg, background=bg, font=("Gothic",12,"bold"), width=tabwidth)
        s.map('TNotebook.Tab', foreground=[('active', 'black')])
        self.pages = []
        if tabnames!=None:
            for i in range(len(tabnames)):
                self.pages.append(ttk.Frame(self))
                self.add(self.pages[i], text=tabnames[i])
        self.pack(fill=tk.BOTH, expand=1)

    def newtab(self, name):
        self.pages.append(ttk.Frame(self))
        self.add(self.pages[-1], text=name)
        return self.pages[-1]

    def getcurrenttab(self):
        print(self.tab(self.select(), "text"))
        return self.pages[self.index(self.select())]

    def getcurrenttabindex(self):
        return self.index(self.select())

class TreeView(tix.CheckList):
    """Tree checklist.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    itemlist : list
        List of item tuples. ([(name1,text1),(name2,text2),(name3,text3)...})
    Attributes
    ----------
    modedict : dict
        Dictionary of item modes. (open close or none)
    """
    def __init__(self, master, itemlist, f_init=lambda self,item,itemarr:self.printinit(item,itemarr), f_select=lambda self,item,itemarr:self.printselected(item,itemarr), indent=20):
        super().__init__(master)
        self.f_init = f_init
        self.f_select = f_select
        self.indent = indent
        self.configure(width=500, height=500, highlightthickness=0, highlightcolor='#000000')
        self.textdict = {}
        #self.modedict = {}
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.font = tk.font.Font(family="Gothic",size=12)
        self.indicator_default = self.hlist["indicatorcmd"]
        self.hlist.configure(browsecmd=self.selectCommand, command=self.selectCommand, indicatorcmd=self.indicatorCommand, selectmode="single", indent=indent, padx=2, pady=2, relief=tix.GROOVE, font=self.font, bg="#ffffff")
        initset = set()
        itemset = set()
        for i in itemlist:
            if isinstance(i[0],(list,tuple)):
                item = TreeView.list2item(i[0])
                self.hlist.add(item, text=i[1], state="disabled")
                self.textdict[item] = i[1]
                itemset.add(item)
                if len(i[0]) == 1:  #list of one element
                    initset.add(item)
            else:
                item = i[0]
                self.hlist.add(item, text=i[1], state="disabled")
                self.textdict[item] = i[1]
                itemset.add(item)
                if item.count(".") == 0:
                    initset.add(item)
        self.autosetmode()
        for i in initset:
            self.initItem(i)
        for i in itemset:
            self.f_init(self,i,TreeView.item2list(i))
        master.update()
        self.enable()
        self.returnflag = False
        self.hlist.bind("<Return>",self.evReturn)
        self.hlist.bind("<ButtonPress-1>",self.evButtonPress)
        self.hlist.bind("<ButtonRelease-1>",self.evButtonRelease)

    def selectCommand(self, item):
        x = self.winfo_pointerx()-self.winfo_rootx()
        y = self.winfo_pointery()-self.winfo_rooty()
        w = self.winfo_width()
        h = self.winfo_height()
        xlen_cbx = 13 + 4                                      #checkbox left padding (2) + checkbox right padding (2) = 4
        xlen_txt = self.font.measure(self.textdict[item]) + 4  #text left padding (2) + text right padding (2) = 4
        xlen_ind = self.getIndent(item) + 1                    #indent
        xlen_scr = self.hlist.xview()[0]                       #scrollbar shift
        xmin = xlen_ind - xlen_scr
        xmax = xlen_ind + xlen_cbx + xlen_txt - xlen_scr
        #print(self.hlist.info_bbox(item))
        ymin = self.hlist.info_bbox(item)[1]
        ymax = self.hlist.info_bbox(item)[3]
        if (xmin <= x and x <= xmax and ymin <= y and y <= ymax) or self.returnflag:
            if self.getstatus(item) == "on":
                self.setstatus(item, "off")
            else:
                self.setstatus(item, "on")
            self.selectItem(item)
        self.returnflag = False
        self.hlist.selection_clear()

    def evReturn(self, event):
        #item = self.hlist.info_selection()[0]
        self.returnflag = True

    def indicatorCommand(self,item):
        if self.indicator_allow:
            if self.getmode(item) == "open":
                self.open(item)
            else:
                self.close(item)
        self.indicator_allow = False
        self.hlist.selection_clear()

    def evButtonPress(self, event):
        self.indicator_allow = True

    def evButtonRelease(self, event):
        self.indicator_allow = False

    def list2item(alist):
        for i in range(len(alist)):
            if alist[i].count(".") > 0:
                alist[i] = alist[i].replace(".","<#DOT>")
        return ".".join(alist)

    def item2list(item):
        alist = item.split(".")
        for i in range(len(alist)):
            if alist[i].count("<#DOT>") > 0:
                alist[i] = alist[i].replace("<#DOT>",".")
        return alist

    def printinit(self, item, itemarr):
        print(" " + TreeView.list2item(itemarr))

    def printselected(self, item, itemarr):
        print(" " + TreeView.list2item(itemarr))

    def enable(self, item=None):
        if item==None:
            if self.hlist.info_children():
                for child in self.hlist.info_children():
                    self.hlist.entryconfigure(child, state="enabled")
                    self.enable(child)
        else:
            if self.hlist.info_children(item):
                for child in self.hlist.info_children(item):
                    self.hlist.entryconfigure(child, state="enabled")
                    self.enable(child)

    def selectAll(self, status="on", item=None):
        if item==None:
            if self.hlist.info_children():
                for child in self.hlist.info_children():
                    self.setstatus(child, status)
                    self.f_select(self,child,TreeView.item2list(child))
                    self.selectAll(status,child)
        else:
            if self.hlist.info_children(item):
                for child in self.hlist.info_children(item):
                    self.setstatus(child, status)
                    self.f_select(self,child,TreeView.item2list(child))
                    self.selectAll(status,child)

    def initItem(self, item):
        self.close(item)
        self.setstatus(item, "off")
        #self.modedict[item] = self.getmode(item)
        if self.hlist.info_children(item):
            for child in self.hlist.info_children(item):
                self.initItem(child)

    def selectItem(self, item):
        #if self.modedict[item] != self.getmode(item):
        #    self.modedict[item] = self.getmode(item)
        #    return
        self.f_select(self,item,TreeView.item2list(item))
        self.selectParent(item)
        self.selectChild(item)
        #print("-----selection-----")
        #print(self.getSelected(1))
        #print(self.getselection("on"))
        #print(item, self.getstatus(item))

    def selectParent(self, item):
        if self.hlist.info_parent(item):
            parent = self.hlist.info_parent(item)
            if(self.getstatus(item) == "on"):
                self.setstatus(parent, "on")
                self.f_select(self,parent,TreeView.item2list(parent))
                self.selectParent(parent)
            elif(self.getstatus(item) == "off"):
                if self.hlist.info_children(parent):
                    for child in self.hlist.info_children(parent):
                        if(self.getstatus(child) == "on"):
                            return
                    self.setstatus(parent, "off")
                    self.f_select(self,parent,TreeView.item2list(parent))
                    self.selectParent(parent)

    def selectChild(self, item):
        if self.hlist.info_children(item):
            for child in self.hlist.info_children(item):
                self.setstatus(child, self.getstatus(item))
                self.f_select(self,child,TreeView.item2list(child))
                self.selectChild(child)

    #h = hierarchy
    def getSelected(self, h=0):
        r = []
        s = self.getselection("on")
        for i in s:
             if(i.count(".") == h):
                  r.append(i)
        return r

    def getIndent(self, item):
        return self.indent*(item.count(".")+1)

class DatePane(ttk.PanedWindow):
    """A pane with date entry.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    text : str
        Text of label.
    fg : str
        Foreground color. (Text color)
    bg : str
        Background color.
    width : int
        Width of textbox.
    Attributes
    ----------
    lbl : ttk.Label
        Label.
    y : ttk.Entry
        Year textbox.
    m : ttk.Entry
        Month textbox.
    d : ttk.Entry
        Day textbox.
    """
    def __init__(self, master, text, fg, bg, fg_enable="black", fg_disable="gray", width=4):
        super().__init__(master)
        self.fg_enable = fg_enable
        self.fg_disable = fg_disable
        self.configure(style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.pack()
        self.lbl = ttk.Label(self, text=text, style=StyleManager().getstyle("TLabel",fg,bg))
        self.lbl.pack(padx=2, side=tk.LEFT)
        self.y = NumberEntry(self, width, "yyyy", 4)
        self.m = NumberEntry(self, width, "mm", 2)
        self.d = NumberEntry(self, width, "dd", 2)
    def enable(self):
        self.lbl.configure(foreground=self.fg_enable)
        self.y["state"] = tk.NORMAL
        self.m["state"] = tk.NORMAL
        self.d["state"] = tk.NORMAL
    def disable(self):
        self.lbl.configure(foreground=self.fg_disable)
        self.y["state"] = tk.DISABLED
        self.m["state"] = tk.DISABLED
        self.d["state"] = tk.DISABLED
    def getdatetime(self):
        try:
            d = datetime.datetime.strptime(self.y.get() + "-" + self.m.get() + "-" + self.d.get(), "%Y-%m-%d")
        except ValueError:
            return None
        return d
    def getdatestr(self,sep="-"):
        d = self.getdatetime()
        if d is None:
            return None
        else:
            return d.strftime(sep.join(["%Y","%m","%d"]))

class RadioSet(ttk.PanedWindow):
    """A set of radio buttons.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    btnlist : list
        List of button names. (str)
    fg : str
        Foreground color. (Text color)
    bg : str
        Background color.
    state : Tk state
        Button state. (NORMAL ACTIVE DISABLED)
    f : function
        Function to run. (Command)
    Attributes
    ----------
    btns : list
        List of radio buttons. (ttk.Radiobutton)
    v : tk.StringVar
        Variable holding current choice.
    """
    def __init__(self, master, btnlist, fg, bg, state=tk.NORMAL, f=None):
        super().__init__(master)
        self.configure(style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.pack()
        if f == None:
            f = self.dummy       #no command; set to dummy function
        self.btns = []
        self.v = tk.StringVar()  #variable holding current choice (set initially to 0)
        self.v.set(0)
        for i, text in enumerate(btnlist):
            self.btns.append(ttk.Radiobutton(self, value=i, text=text, variable=self.v, state=state, command=f, style=StyleManager().getstyle("TRadiobutton",fg,bg)))
            #self.btns[i].config(indicatoron=False)  #for tk.RadioButton
            self.btns[i].pack(anchor=tk.W)
    def dummy(self):
        pass
    def enable(self):
        """Enable radio buttons.
        """
        for b in self.btns:
            b.config(state=tk.NORMAL)
    def disable(self):
        """Disable radio buttons.
        """
        for b in self.btns:
            b.config(state=tk.DISABLED)
    def setcmd(self,f):
        """Set command.
        Parameters
        ----------
        f : function
            Function to run. (Command)
        """
        for b in self.btns:
            b.configure(command=f)
    def choice(self):
        """
        Returns
        -------
        str
            Current choice as text.
        """
        i = int(self.v.get())
        b = self.btns[i]
        t = b["text"]
        #print(t)
        return t

class ImageLabel(ttk.Label):
    """A label with an image.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    file : str
        File path of image. (gif)
    padding : int
        Border padding.
    Attributes
    ----------
    image : tk.PhotoImage
        Photo image object.
    """
    def __init__(self, master, file, padding=0, fg="white", bg="white"):
        super().__init__(master)
        photo = tk.PhotoImage(file=file)
        self.configure(image=photo, style=StyleManager().getstyle("TLabel",fg,bg))
        self.image = photo
        self.pack(padx=padding,pady=padding)

class ImagePane(ttk.PanedWindow):
    """A paned window with an image label.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    file : str
        File path of image. (gif)
    Attributes
    ----------
    image : tk.PhotoImage
        Photo image object.
    lpane : ttk.PanedWindow
        Left pane. Pane left of the image.
    rpane : ttk.PanedWindow
        Right pane. Pane right of the image.
    """
    def __init__(self, master, file, fg="white", bg="white"):
        super().__init__(master)
        style_p = StyleManager().getstyle("TPanedwindow",fg,bg)
        self.configure(style=style_p)
        self.lpane = ttk.PanedWindow(self, width=20, style=style_p)
        self.lpane.pack(side=tk.LEFT)
        self.image = ImageLabel(self,file)
        self.image.pack(side=tk.LEFT)
        self.rpane = ttk.PanedWindow(self, style=style_p)
        self.rpane.pack(side=tk.RIGHT)

class LabelPane(ttk.PanedWindow):
    """A paned window with a label.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    text : str
        Text of label.
    fg : str
        Foreground color. (Text color)
    bg : str
        Background color.
    Attributes
    ----------
    lbl : ttk.Label
        Top label.
    botpane : ttk.PanedWindow
        Bottom pane. Pane below the top label.
    paddingpane : ttk.PanedWindow
        Extra padding to the left of the bottom pane.
    """
    def __init__(self, master, text, fg, bg):
        super().__init__(master)
        style_p = StyleManager().getstyle("TPanedwindow",fg,bg)
        self.lbl = ttk.Label(self, text=text, style=StyleManager().getstyle("TLabel",fg,bg))
        self.lbl.pack(anchor=tk.N, side=tk.TOP, fill=tk.X, expand=1)
        self.botpane = ttk.PanedWindow(self, style=style_p)
        self.botpane.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.paddingpane = ttk.PanedWindow(self.botpane, width=20, style=style_p)
        self.paddingpane.pack(side=tk.LEFT, fill=tk.Y)
        self.configure(style=style_p)
        self.pack(padx=5, fill=tk.BOTH, expand=1)

class Meter(ttk.PanedWindow):
    """A progress bar with label.
    Parameters
    ----------
    master : Tk object
        Tk object to use as master.
    text : str
        Text of label.
    fg : str
        Foreground color. (Text color)
    bg : str
        Background color.
    orient : str
        Horizontal or vertical orientation.
    length : int
        Length of progress bar.
    mode : str
        Determinate or indeterminate mode. (Progress bar is updated under program control in determinate mode)
    Attributes
    ----------
    bar : ttk.Progressbar
        Progressbar.
    lbl : ttk.Label
        Label.
    """
    def __init__(self, master, text="", fg="black", bg="white", orient="horizontal", length=300, mode="determinate"):
        super().__init__(master)
        self.configure(style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.pack()
        self.bar = ttk.Progressbar(self, orient=orient, length=length, mode=mode, value=0, maximum=100)
        self.bar.pack(anchor=tk.W, side=tk.TOP)
        self.lbl = ttk.Label(self, text=text, style=StyleManager().getstyle("TLabel",fg,bg))
        self.lbl.pack(anchor=tk.W, side=tk.BOTTOM)

    def update_bar(self, value):
        """Update progressbar.
        """
        self.bar["value"] = value
        self.bar.update()

    def update_lbl(self, text):
        """Update label.
        """
        self.lbl["text"] = text
        self.lbl.update()

class Slider(ttk.PanedWindow):
    def __init__(self, master, fg="black", bg="white", length=200, orient="horizontal", power=-2, from_=0.5, to=1.5, value=None, text=""):
        super().__init__(master)
        self.power = power
        self.from_ = round(float(from_),-1*self.power)
        self.to = round(float(to),-1*self.power)
        if value == None:
            self.value = self.from_
        else:
            self.value = round(float(value),-1*self.power)
            if self.value < self.from_:
                self.value = self.from_
            elif self.value > self.to:
                self.value = self.to
            if self.power >= 0:
                self.value = int(self.value)
        self.inc = 10**self.power
        self.configure(style=StyleManager().getstyle("TPanedwindow",fg,bg))
        self.pack()
        self.lbl_l = ttk.Label(self, width=5, style=StyleManager().getstyle("TLabel",fg,bg))
        self.lbl_l["text"] = text
        self.lbl_l.pack(side=tk.LEFT)
        if orient == "horizontal":
            style=StyleManager().getstyle("Horizontal.TScale",fg,bg)
        elif orient == "vertical":
            style=StyleManager().getstyle("Vertical.TScale",fg,bg)
        self.scl = ttk.Scale(self, from_=self.from_, to=self.to, value=self.value, length=length, orient=orient, style=style, command=self.update)
        self.flag_left = False
        self.flag_right = False
        self.scl.bind("<Left>", self.left)
        self.scl.bind("<Right>", self.right)
        self.scl.pack(side=tk.LEFT)
        self.lbl_r = ttk.Label(self, width=5, style=StyleManager().getstyle("TLabel",fg,bg))
        self.lbl_r["text"] = self.value
        self.lbl_r.pack(side=tk.LEFT)

    #update is called when self.scl.set is called
    def update(self,value):
        if self.flag_left:
            self.flag_left = False
            self.scl.set(self.value-self.inc)
        elif self.flag_right:
            self.flag_right = False
            self.scl.set(self.value+self.inc)
        else:
            self.scl.focus_set()
            self.value = round(float(value),-1*self.power)
            if self.power >= 0:
                self.value = int(self.value)
            self.lbl_r["text"] = self.value
            self.lbl_r.update()

    def left(self,event):
        self.flag_left = True

    def right(self,event):
        self.flag_right = True

    def enable(self):
        self.scl.state(["!disabled"])

    def disable(self):
        self.scl.state(["disabled"])

class Limitedstring():
    def __init__(self,n):
       self.n = n
       self.q = [""]*n
       self.i = n

    def insert(self,x):
       self.i = self.i + 1
       if self.i >= self.n:
           self.i = 0
       self.q[self.i] = x

    def get(self):
       r = ""
       for k in range(self.i+1,self.i+self.n+1):
           r = r + self.q[k%self.n]
       return r

class DebugPane(ttk.PanedWindow):
    def __init__(self, master, f_debug):
        super().__init__(master)
        self.bind_all("<Key>", self.key)
        #self.bind("<Button-1>", self.callback)
        self.pw = Limitedstring(9)
        self.debugmode = False
        self.f_debug = f_debug

    def key(self,event):
        if self.debugmode:
            return
        self.pw.insert(str(event.char))
        if "debugmode" in self.pw.get():
            self.debugmode = True
            self.configure(style=StyleManager().getstyle("TPanedwindow",fg="green",bg="green"))
            print("debug mode on")
            self.f_debug()

    def callback(self,event):
        print("clicked at", event.x, event.y)

def testradioset(rset):
    c = []
    for r in rset:
        c.append(r.choice())
    print(c)

def testdatepane(dp):
    print(dp.getdatestr())

def test():
    print("this is a test")

def createnewwindow1(nwb):
    mbar = Menubar(nwb.root,nwb.on_close)
    tab = TabView(nwb.root, "black", "white", ["a","b","c","d","e","f","g","h"], 5)
    itemlist1 = [("CL1","checklist1"),("CL1.Item1","subitem1"),("CL1.Item1.Item1","subsubitem1"),("CL1.Item1.Item2","subsubitem2"),("CL1.Item2","subitem2"),("CL1.Item2.Item1","subsubitem1"),("CL1.Item2.Item2","subsubitem2"),("CL2","checklist2"),("CL2.Item1","subitem1")]
    tree1 = TreeView(tab.pages[0],itemlist1,f_init=lambda tree,item,itemarr:tree_init(tree,item,itemarr),f_select=lambda tree,item,itemarr:tree_select(tree,item,itemarr))
    itemlist2 = [("1","1"),("1.2","1.2"),("1.2.3","1.2.3"),("1.2.3.4","1.2.3.4"),("1.2.3.4.5","1.2.3.4.5"),(["1",".test."],".test."),(["..."],"...")]
    tree2 = TreeView(tab.pages[1],itemlist2,f_init=lambda tree,item,itemarr:tree_init(tree,item,itemarr),f_select=lambda tree,item,itemarr:tree_select(tree,item,itemarr))

def createnewwindow2(root):
    PlotCanvas(root)

def sub_process(x):
    return x

def main_process():
    print("main")

def tree_init(tree,item,itemarr):
    print("   " + item)
    print("   " + TreeView.list2item(itemarr))

def tree_select(tree,item,itemarr):
    print("   " + item)
    print("   " + TreeView.list2item(itemarr))

def createdialog(root):
    d = Dialogbox(root,"test","\n123\n\n456\nabcdefghijklmnopqrstuvwxyz",image="C:\\Users\\wcp\\Desktop\\kpi\\main\\i2.gif")
    d.inserttext("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",link1)
    d.inserttext("\n\n")
    d.inserttext("2",link2)
    d.inserttext("\n")
    d.inserttext("3",link3)
    d.inserttext("\n")
    d.inserttext("4",link4)
    d.inserttext("\n\n")
    d.inserttext("1",link1)
    d.inserttext("\n\n")
    d.insertend()
    d.centerwindow()

def link1():
    print("link1")
    Filemngr.opendir("C:\\")
    Filemngr.opennote("C:\\Users\\wcp\\Desktop\\kpi\\main\\example.log")

def link2():
    print("link2")

def link3():
    print("link3")

def link4():
    print("link4")

if __name__ == '__main__':
    root = tix.Tk()
    de = DirEntry(root,"test","white","black")
    l = LabelPane(root,"test","black","white")
    cc = ColorCanvas(l.botpane)
    cc.pack(side=tk.TOP)
    sb = SinkButton(l.botpane, "sink")
    sb.pack(side=tk.TOP)
    slider1 = Slider(l.botpane,power=-2,from_=1.0,to=2.0,value=1.05,text="abcde")
    slider2 = Slider(l.botpane,power=0,from_=50,to=1000,value=100,text="fghij")
    m = Meter(l.botpane,"progress...","black","white")
    r1 = RadioSet(l.botpane,["1","2","3"],"white","green")
    #r1.setcmd(r1.choice)
    r1.pack(side=tk.LEFT)
    r2 = RadioSet(l.botpane,["4","5","6"],"white","green")
    #r2.disable()
    r2.pack(side=tk.LEFT)
    r3 = RadioSet(l.botpane,["7","8","9"],"white","green")
    r3.pack(side=tk.LEFT)
    d1 = DatePane(l.botpane,"start","white","blue")
    #i1 = ImageLabel(l.botpane,"C:\\Users\\wcp\\Desktop\\node.gif")
    pgrid = ttk.PanedWindow(l.botpane)
    pgrid.pack(side=tk.LEFT)
    b1 = ttk.Button(pgrid, text="RadioSet", command = lambda: testradioset([r1,r2,r3]))
    b2 = ttk.Button(pgrid, text="DatePane", command = lambda: testdatepane(d1))
    b3 = ttk.Button(pgrid, text="Thread", command = lambda: gui.thread_stop())
    b4 = ttk.Button(pgrid, text="Plot", command = lambda: createnewwindow2(root))
    b5 = ttk.Button(pgrid, text="Dialog", command = lambda: createdialog(root))
    b6 = ttk.Button(pgrid, text="Slider", command = lambda: print(slider1.value,slider2.value))
    b1.grid(row=0,column=0)
    b2.grid(row=0,column=1)
    b3.grid(row=0,column=2)
    b4.grid(row=1,column=0)
    b5.grid(row=1,column=1)
    b6.grid(row=1,column=2)
    n1 = NewWindowButton(l.botpane,"Create")
    n1.f_create = lambda: createnewwindow1(n1)
    n2 = ToggleButton(l.botpane)
    n1.pack(side=tk.LEFT)
    n2.pack(side=tk.LEFT)
    gui = ThreadedGUI(root)
    for i in range(10):
        gui.thread_start(lambda x=i: sub_process(x),main_process)
    gui.thread_start(cc.color)
    for i in range(10):
        gui.thread_start(lambda x=i: sub_process(x),main_process)
    f = lambda: test()
    f()
    m.after(5000, lambda:[m.update_bar(50),m.update_lbl("testing meter(50)")])
    importlib.import_module("tensorflow")
    #mp1 = MProc()
    #mp1.test()
    print(os.cpu_count())
    gui.disable()
    root.mainloop()
