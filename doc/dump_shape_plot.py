import pandas
import matplotlib.pyplot as plt
import sys

table = pandas.read_csv(sys.stdin, sep = '\t')
table.plot(x = 'x', y = 'shape', grid = True)
plt.show()
