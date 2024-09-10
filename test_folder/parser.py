import pandas as pd
import numpy as np  

data = pd.read_excel('/Users/rushirbhavsar/Downloads/2022-2017-NAICS-Code-Concordance-1.xlsx')

codes = data['2022 NAICS Code'].values

array = '['

for code in codes:
    array += str(code) + ', '

array += ']'

print(array)