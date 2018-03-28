#configparser wrapper

import configparser
import re
import os

class INIParser():

    def __init__(self, inipath=None):
        self.cp = configparser.SafeConfigParser()
        if inipath != None:
            self.read(inipath)

    def read(self, inipath):
        self.cp.read(inipath)

    def write(self, outpath):
        with open(outpath, "w") as inifile:
            self.cp.write(inifile)

    def get(self, section, key):
        return self.cp.get(section, key)

    def set(self, section, key, value):
        try:
            self.cp.add_section(section)
        except configparser.DuplicateSectionError:
            pass
            #print("duplicate section: " + section)
        finally:
            self.cp.set(section, str(key), str(value))

    def has_section(self, section):
        return self.cp.has_section(section)

    def has_option(self, section, key):
        return self.cp.has_option(section, key)

    def remove_section(self, section):
        return self.cp.remove_section(section)

    def remove_option(self, section, key):
        return self.cp.remove_option(section, key)

    def remove_option_all(self, key):
        for section in self.cp.sections():
            self.remove_option(section, key)

    def print(self):
        for section in self.cp.sections():
            print("["+section+"]")
            for key in self.cp.options(section):
                print(key + ": " + self.get(section,key))

    #escape % symbols in ini file
    def esc_percent(inipath,writepath):
        lines = list(open(inipath))
        for line in lines:
            if "%" in line:
                line = line.strip()
                if line[0] == "[" and line[-1] == "]":
                    pass
                else:
                    line = line.replace("%%","%")
                    line = line.replace("%","%%")
            else:
                pass
        with open(writepath, "w") as f:
            for line in lines:
                f.write(line)

if __name__ == '__main__':
    ip = INIParser("ini_test.ini")
    ip.set("section1","key1",1)
    ip.set("section1","key2",2)
    ip.set("section2","key1",3)
    ip.set("section2","key2",4)
    print(ip.get("section1","key1"))
    print(ip.get("section2","key1"))
    ip.remove_option_all("key1")
    ip.print()
    #ip.write("ini_test_new.ini")
    #INIParser.esc_percent("ini_test.ini","ini_test_esc.ini")
