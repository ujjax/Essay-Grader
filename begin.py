from __future__ import division

import string
import math

import numpy as np
import os
import sys
import pandas as pd
import csv
import re
import collections

tokenize = lambda doc: doc.lower().split(" ")


def createdata(doc):
    words = " ".join(doc).split()
    count = collections.Counter(words).most_common()
    rdic = [i[0] for i in count]
    dic = {w: i for i, w in enumerate(rdic)}
    voc_size = len(rdic)
    arrdoc = np.asarray(doc)
    final = []
    for i in range(arrdoc.shape[0]):
        words = arrdoc[i].split()
        val = []
        for word in words:
            val.append(dic[word])
        final.append(val)

    return final


def cleanstring(s):
    s = s.lower()
    s = s.replace(".", "")
    s = s.replace("\\", "")
    s = s.replace("\"", "")
    s = s.replace("'", "")
    s = s.replace(",", "")
    s = s.replace(":", "")
    s = s.replace(";", "")
    s = s.replace("-", "")
    s = s.replace("?", "")
    s = s.replace("!", "")
    s = s.replace("#", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("/", " ")
    s = ' '.join([word for word in s.split() if not word.startswith('@')])
    return s


def finaldata(data):
    documents = []
    data = np.asarray(data)
    for i in range(len(data)):
        str = data[i, 0]
        documents.append(cleanstring(str))
    xdata = tuple(createdata(documents))
    ydata = tuple(list(data[:,1]))
    final_data = zip(xdata,ydata)
    return np.asarray(final_data)



