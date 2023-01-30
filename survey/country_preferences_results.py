import glob
import json
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import matplotlib
import csv

from translators.server import google

file_names = glob.glob("data-all/*.json")                                   # retrieve a list of files from pathname arg
counts = Counter([])                                                        # counter > dict subclass for counting hashable obj
freq = Counter([])
for file_name in file_names:
    data = json.load(open(file_name))
    c = Counter(e['selected'] for e in data['countryPairs'])                # counts how many times a country was selected (given a choice of two countries)
    f = Counter(sum([e["options"] for e in data['countryPairs']], []))      # counts how many times a country appeared among the two-country choices
    counts += c
    freq += f

res = [(n, v/freq[n]) for n, v in counts.items()]
res.sort(key=itemgetter(1), reverse=True)

res_eng = []
for name, val in res:
    eng_name = google(name, from_language='sl', to_language='en')
    print(f'{name} --> {eng_name}')
    res_eng.append((eng_name, val))


print(res_eng)

with open('out.csv', 'w', newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(res_eng)

values = [r[1] for r in res]
names = [r[0] for r in res]
plt.cla()
matplotlib.rc('ytick', labelsize=9)
# matplotlib.rcParams['ytick.major.pad'] = '20'
fig, ax = plt.subplots()
ax.yaxis.labelpad = 40
fig.set_figheight(10)
y_pos = np.arange(len(res))
ax.barh(y_pos, values)
ax.set_yticks(y_pos, labels=names)
ax.invert_yaxis()
plt.savefig("tmp.pdf")
