import pandas as pd
d = pd.read_excel('reports/error_review.xlsx', sheet_name='sentences')
d = d[d.verdict.notna()]
print(d.groupby(['kind', 'verdict']).size())