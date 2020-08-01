import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

## Use this to read data from the csv file on local system.
df = pd.read_csv('./data/retail_data.csv', sep=',') 
## Print top 5 rows 
df.head(5)

items = (df['0'].unique())
items

encoded_vals = []
def custom():
for index, row in df.iterrows():
    labels = {}
    uncommons = list(set(items) - set(row))
    commons = list(set(items).intersection(row))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)

apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False)

freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
freq_items.head(7)

rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
rules.head()

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules[‘support’], rules[‘lift’], alpha=0.5)
plt.xlabel(‘support’)
plt.ylabel(‘lift’)
plt.title(‘Support vs Lift’)
plt.show()

fit = np.polyfit(rules[‘lift’], rules[‘confidence’], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules[‘lift’], rules[‘confidence’], ‘yo’, rules[‘lift’], 
 fit_fn(rules[‘lift’]))
 
 