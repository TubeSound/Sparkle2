# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 20:24:47 2022

@author: IKU-Trader
"""

import os
from pandas import DataFrame
import numpy as np
from time_utils import TimeUtils

import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class Utils:

    @staticmethod
    def makeDir(dirpath: str):
        """ディレクトリを作成する

        Args:
            dirpath (str): ディレクトリパス
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    
    @staticmethod
    def makeDirs(parent_path: str, holders: str):
        """複数のディレクトリを作成する

        Args:
            parent_path (str): フォルダを作成する場所のディレクトリパス
            holders (list): フォルダ名のリスト
        """
        for holder in holders:
            path = os.path.join(parent_path, holder)
            Utils.makeDir(path)

    @staticmethod
    def fileList(dir_path: str, extension: str):
        return glob.glob(os.path.join(dir_path, extension))
        
        
    @staticmethod
    def df2dic(df: DataFrame, is_numpy=False, time_key='time', convert_keys=None):
        columns = df.columns
        dic = {}
        for column in columns:
            d = None
            if column.lower() == time_key.lower():
                nptime = df[column].values
                pytime = [TimeUtils.npDateTime2pyDatetime(t) for t in nptime]
                if is_numpy:
                    d = nptime
                else:
                    d = pytime
            else:
                d = df[column].values.tolist()
                d = [float(v) for v in d]
            if is_numpy:
                d = np.array(d)
            else:
                d = list(d)
            if convert_keys is None:
                key = column
            else:
                try:
                    key = convert_keys[column]
                except Exception as e:
                    key = column
            dic[key] = d
        return dic

    @staticmethod
    def dic2df(dic):
        keys = list(dic.keys())
        values = list(dic.values())
        length = []
        for value in values:
            n = len(value)
            length.append(n)
        if(min(length) != max(length)):
            return None
        out = []
        for i in range(n):
            d = []
            for j in range(len(values)):
                d.append(values[j][i])
            out.append(d)
        df = DataFrame(out, columns=keys)
        return df

    @staticmethod
    def splitDic(dic, i):
        keys = dic.keys()
        arrays = []
        for key in keys:
            arrays.append(dic[key])
        split1 = {}
        split2 = {}
        for key, array in zip(keys, arrays):
            split1[key] = array[:i]
            split2[key] = array[i:]
        return (split1, split2)

    @staticmethod    
    def deleteLast(dic):
        keys = dic.keys()
        arrays = []
        for key in keys:
            arrays.append(dic[key])
        out = {}
        for key, array in zip(keys, arrays):
            out[key] = array[:-1]
        return out        
    
    @staticmethod            
    def dic2Arrays(dic):
        keys = dic.keys()
        arrays = []
        for key in keys:
            arrays.append(dic[key])
        return keys, arrays
    
    @staticmethod    
    def array2Dic(array, keys):
        dic = {}
        for key, i in enumerate(keys):
            d = []
            for a in array:
                d.append(a[i])
            dic[key] = d
        return dic
     
    @staticmethod        
    def insertDicArray(dic: dict, add_dic: dict):
        keys = dic.keys()
        try:
            for key in keys:
                a = dic[key]
                a += add_dic[key]
            return True
        except:
            return False


    @staticmethod        
    def findTime(pytime_array: list, time, length):
        index = None
        for i, t in enumerate(pytime_array):
            if t >= time:
                index = i
                break
        if index is None:
            return (None, None, None)
        if index == 0:
            return (None, None, None)
        begin = index -  length
        if begin < 0:
            begin = 0
        end = index + length
        if end >= len(pytime_array):
            end = len(pytime_array) - 1
        return (int(begin), int(index), int(end)) 
        
    @staticmethod                
    def sliceBetween(data: dict, pytime_array: list, time_from, time_to):
        n, begin, end = TimeUtils.sliceTime(pytime_array, time_from, time_to)
        if n == 0:
            return (0, None)
        d = Utils.sliceDict(data, begin, end)
        a = d[list(d.keys())[0]]
        return len(a), d
        
    @staticmethod        
    def sliceDict(dic, begin, end):
        keys = list(dic.keys())
        arrays = []
        for key in keys:
            arrays.append(dic[key])
        out = {}
        for key, array in zip(keys, arrays):
            out[key] = array[begin: end + 1]
        return out
    
    @staticmethod
    def sliceDictLast(dic, size):
        keys = list(dic.keys())
        n = len(dic[keys[0]])
        begin = n - size
        if begin < 0:
            begin = 0
        return Utils.sliceDict(dic, begin, n - 1)
        
    @staticmethod
    def sliceDictWithKeys(dic, begin, end, keys):
        out = {}
        for key in keys:
            d = dic[key]
            out[key] = d[begin: end + 1]
        return out
    
    @staticmethod
    def sliceDict2Array(dic, begin, end, keys):
        arrays = []
        for key in keys:
            d = dic[key]
            arrays.append(d[begin: end + 1])
        return arrays
    
    @staticmethod 
    def saveArrays(filepath, arrays):
        l = None
        for array in arrays:
            if l == None:
                l = len(array)
            else:
                if l != len(array):
                    print('Bad array length')
                    return       
        f = open(filepath, mode='w', encoding='sjis')
        for j in range(l):
            s = ''
            for array in arrays:
                s += str(array[j]) + ','
            s = s[:-1]
            f.write(s + '\n')
        f.close()
        
    @staticmethod
    def fig_html(fig):
        header = '<!doctype html><html lang="ja"><body>'
        template = '<img src="data:image/png;base64,{image_bin}">'
        fotter = '</body></html>'
        sio = io.BytesIO()
        fig.savefig(sio, format='png')
        image_bin = base64.b64encode(sio.getvalue())
        image_html = template.format(image_bin=str(image_bin)[2:-1])
        return header, fotter, image_html 
