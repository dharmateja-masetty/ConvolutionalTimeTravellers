# import numpy as nup
# def qcd_variance(series:any):
#         window=12
#         variances = series.rolling(window).var().dropna()
#         Q1 = nup.percentile(variances, 25, interpolation='midpoint')
#         Q3 = nup.percentile(variances, 75, interpolation='midpoint')
#         qcd = round((Q3 - Q1)/(Q3 + Q1),6)
#         return (qcd)