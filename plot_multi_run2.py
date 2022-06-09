from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from config import multi_run_results_file_path, MAX_CASE

filename = multi_run_results_file_path


colNames = pd.read_csv(filename, nrows=1).columns
dataset = pd.read_csv(filename, usecols=colNames)

print(dataset)

tau10 = dataset.loc[dataset['tau_setup'] == '-1']
tau10 = tau10.loc[tau10['case'] == '1']
tau10 = tau10.dropna(axis=1, how='all')
tau10 = tau10[['tau_setup','lossValue']]

print(tau10)

fig,ax = plt.subplots()
ax.plot(tau10['lossValue'], color="red")
# ax.set_xlabel('Time Value (s)')
# ax.set_ylabel('Loss Value', color="red")
# ax2 = ax.twinx()
# ax2.plot(tau10['tValue'], tau10['deltaAdapt'], color="blue")
# ax2.set_ylabel('Delta Value', color="blue")
plt.show()