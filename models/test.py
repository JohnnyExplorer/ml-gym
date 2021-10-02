
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('./datasets/diamonds.csv', index_col=0)
print(data.head())
