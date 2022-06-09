from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from config import single_run_results_file_path, MAX_CASE

filename = '/Users/udaiarneja/OneDrive - Imperial College London/Work/Year 4/Final Year Project/adaptive-federated-learning/results/StepSize0.00001.csv'
filename2 = '/Users/udaiarneja/OneDrive - Imperial College London/Work/Year 4/Final Year Project/adaptive-federated-learning/results/StepSize0.01.csv'
filename3 = '/Users/udaiarneja/OneDrive - Imperial College London/Work/Year 4/Final Year Project/adaptive-federated-learning/results/StepSize0.1.csv'
filename4 = '/Users/udaiarneja/OneDrive - Imperial College London/Work/Year 4/Final Year Project/adaptive-federated-learning/results/StepSize0.001.csv'
filename5 = '/Users/udaiarneja/OneDrive - Imperial College London/Work/Year 4/Final Year Project/adaptive-federated-learning/results/StepSize0.05.csv'
filename6 = '/Users/udaiarneja/OneDrive - Imperial College London/Work/Year 4/Final Year Project/adaptive-federated-learning/results/StepSize100.csv'

df = pd.read_csv (filename)
df2 = pd.read_csv (filename2)
df3 = pd.read_csv (filename3)
df4 = pd.read_csv (filename4)
df5 = pd.read_csv (filename5)
df6 = pd.read_csv (filename6)


fig,ax = plt.subplots()
ax.plot(df['tValue'], df['lossValue'], color="red")
ax.set_xlabel('Time Value (s)')
ax.set_ylabel('Loss Value', color="red")
ax.plot(df2['tValue'], df2['lossValue'], color="blue")
ax.set_xlabel('Time Value (s)')
ax.set_ylabel('Loss Value', color="red")
ax.plot(df3['tValue'], df3['lossValue'], color="red")
ax.set_xlabel('Time Value (s)')
ax.set_ylabel('Loss Value', color="red")
ax.plot(df4['tValue'], df4['lossValue'], color="red")
ax.set_xlabel('Time Value (s)')
ax.set_ylabel('Loss Value', color="red")
ax.plot(df5['tValue'], df5['lossValue'], color="red")
ax.set_xlabel('Time Value (s)')
ax.set_ylabel('Loss Value', color="red")
# ax.plot(df6['tValue'], df6['lossValue'], color="red")
# ax.set_xlabel('Time Value (s)')
# ax.set_ylabel('Loss Value', color="red")
plt.show()