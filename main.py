import numpy as np
import pandas as pd
import camelot
import matplotlib.pyplot as plt
from scipy import stats

# extracting matrix from pdf covnerting into csv
tables = camelot.read_pdf('first homework_UFAZ.pdf',
                          pages='5', flavor='stream')
tables[0].to_csv('matrix.csv')

# creating Matrix_A
Matrix_A = pd.read_csv('matrix.csv', index_col=0).squeeze("columns")
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
Matrix_A.columns = [months]
Matrix_A.index.name = None

#showing the data
print(Matrix_A)

# sorting all data into Matrix_B
Matrix_B = {}
values = []
years = list(Matrix_A.index.values)
for i in years:
    for y in months:
        Matrix_B[str(str(i)+' '+y)] = Matrix_A.loc[i][y]
        values.append(Matrix_A.loc[i][y])

for i in range(3):
    #print first 3 rows of Matrix B
    print(*list(Matrix_B.items())[i])

# plotting data
plt.figure()
plt.subplot(3, 3, 1)
lists = sorted(Matrix_B.items())
x, y = zip(*lists)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.plot(x, y, color='#271AE5', label="Data")
Matrix_B = Matrix_A
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper right', prop={'size': 6})
plt.xlabel('Months')
plt.ylabel('Price dollars per million BTU')


# сalculating mean and standart deviation 1,2,3
std1 = np.std(values)
std2 = 2*std1
std3 = 3*std1
mean = np.mean(values)

# plotting mean +- std1,2,3
print("The mean value of all months from 1997 to 2021: {}\n\nThe standart deviation values:\nstd1: {}\nstd2: {}\nstd3: {}\n".format(mean, std1, std2, std3))
plt.subplot(3, 3, 2)
x, y = zip(*lists)
plt.plot(x, y, color='#271AE5')
plt.axhline(y=mean, color='#1D9B07', linestyle='-', label='Mean')
plt.axhline(y=mean+std1, color='#5CBE4B',
            linestyle='-', label='Standart deviation 1')
plt.axhline(y=mean-std1, color='#5CBE4B', linestyle='-')
plt.axhline(y=mean-std2, color='#97D68C',
            linestyle='-', label='Standart deviation 2')
plt.axhline(y=mean+std2, color='#97D68C', linestyle='-')
plt.axhline(y=mean+std3, color='#C9FEC0',
            linestyle='-', label='Standart deviation 3')
plt.fill_between(x, mean+std1, mean-std1, color='#1D9B07', alpha=0.5)
plt.fill_between(x, mean+std2, mean-std2, color='#1D9B07', alpha=0.3)
plt.fill_between(x, mean+std3, mean-std3, color='#1D9B07', alpha=0.1)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper right', prop={'size': 6})
plt.xlabel('Months')
plt.ylabel('Price dollars per million BTU')

# counting precentage of samples and checking for gaussian distribution
std1count, std2count, std3count = 0, 0, 0
for i in years:
    for y in months:

        if mean-std3 < Matrix_B.loc[i][y] < mean+std3:
            std3count += 1
        if mean-std2 < Matrix_B.loc[i][y] < mean+std2:
            std2count += 1
        if mean-std1 < Matrix_B.loc[i][y] < mean+std1:
            std1count += 1
percent1 = (std1count/300)*100
percent2 = (std2count/300)*100
percent3 = (std3count/300)*100
print("{}% of samples are in range of std1\n{}% of samples are in range of std2\n{}% of samples are in range of std3\n".format(
    percent1, percent2, percent3))

# quartiles
values_init = values
values = sorted(values)
array = np.array(values)


def find_median(array_sorted):
    indices = []
    list_size = len(array_sorted)
    median = 0
    if list_size % 2 == 0:
        # -1 because index starts from 0
        indices.append(int(list_size / 2) - 1)
        indices.append(int(list_size / 2))
        median = (array_sorted[indices[0]] + array_sorted[indices[1]]) / 2
        pass
    else:
        indices.append(int(list_size / 2))
        median = array_sorted[indices[0]]
        pass
    return median, indices
    pass


median, median_indices = find_median(values)
Q1, Q1_indices = find_median(values[:median_indices[0]])
Q3, Q3_indices = find_median(values[median_indices[-1] + 1:])
IQR = Q3-Q1
print("Q1:{}\nmedian:{}\nQ3:{}\nInterquartile range:{}\n".format(Q1, median, Q3, IQR))

# plotting q1,q3,mean
plt.subplot(3, 3, 3)
x, y = zip(*lists)
plt.plot(x, y, color='#271AE5')
plt.axhline(y=Q1, color='#CAC414', linestyle='-', label='Q1')
plt.axhline(y=median, color='#22CA14', linestyle='-', label='Median')
plt.axhline(y=Q3, color='#14B4CA', linestyle='-', label='Q3')
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper right', prop={'size': 6})
plt.xlabel('Months')
plt.ylabel('Price dollars per million BTU')

# q1,q1 percentages
q1count, q3count, q1q3between = 0, 0, 0
for i in years:
    for y in months:
        if Matrix_B.loc[i][y] < Q1:
            q1count += 1
        if Q1 < Matrix_B.loc[i][y] < Q3:
            q1q3between += 1
        if Q3 < Matrix_B.loc[i][y]:
            q3count += 1
percentq1 = (q1count/300)*100
percentq1q3 = (q1q3between/300)*100
percentq3 = (q3count/300)*100
print("{} % of values are below q1\n{} % of values are between q1 and q3\n{}% of values are above q3\n".format(
    percentq1, percentq1q3, percentq3))


# moving mean,std1,std2,std3
plt.subplot(3, 3, 4)
numbers_series = pd.Series(values_init)
moving_averages = round(numbers_series.ewm(alpha=0.5, adjust=False).mean(), 2)
moving_averages_list = moving_averages.tolist()
plt.plot(values_init, color='#271AE5', label='Data')
numbers_series.rolling(window=36).mean().plot(style='r', label='Moving mean')
numbers_series.rolling(window=36).std().plot(
    style='#5CBE4B', label='Moving Std1')
(numbers_series.rolling(window=36).std() *
 2).plot(style='#97D68C', label='Moving Std2')
(numbers_series.rolling(window=36).std() *
 3).plot(style='#C9FEC0', label='Moving Std3')
plt.grid(linestyle=':')
plt.legend(loc='upper right', prop={'size': 6})
plt.xlabel('Months')
plt.ylabel('Price dollars per million BTU')

# moving q1,q3,min,max, median
plt.subplot(3, 3, 5)
plt.plot(values_init, color='#271AE5', label='Data')
numbers_series.rolling(window=36, min_periods=1).quantile(
    0.25, interpolation='higher').plot(style='#CAC414', label='Moving Q1')
numbers_series.rolling(window=36, min_periods=1).quantile(
    0.75, interpolation='lower').plot(style='#14B4CA', label='Moving Q3')
numbers_series.rolling(window=36).min().plot(
    style='#3F3F3F', label='Moving minimum')
numbers_series.rolling(window=36).max().plot(
    style='#FF0000', label='Moving maximum')
numbers_series.rolling(window=36).median().plot(
    style='#22CA14', label='Moving median')
plt.grid(linestyle=':')
plt.legend(loc='upper right', prop={'size': 6})
plt.xlabel('Months')
plt.ylabel('Price dollars per million BTU')

# histogramm
plt.subplot(3, 3, 6)
thickness = int(np.sqrt(len(values_init)))
plt.xlabel("Price dollars per million BTU")
plt.ylabel("Frequency")
mode = numbers_series.mode()
plt.hist(values_init, bins=thickness, color='#42C832',
         edgecolor='#2F8525', linewidth=0.5)
plt.plot(mode, [74, 74], 'ro', label='Modes (2.98,2.99)')
plt.legend(loc='upper right', prop={'size': 6})
plt.grid(linestyle=':')

# boxplot
plt.subplot(3, 3, 7)
bp = plt.boxplot(numbers_series, patch_artist=True)
for box in bp['boxes']:
    box.set(color='#7570b3', linewidth=1)
    box.set(facecolor='#1b9e77')
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=1)
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=1)
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=1, label='Median')
for flier in bp['fliers']:
    flier.set(marker='x', color='#2F8525', alpha=0.5, label='Outliers')
plt.ylabel("Price dollars per million BTU")
plt.xlabel("Time interval (1997-2021)")
plt.grid(linestyle=':')
plt.legend(loc='upper right', prop={'size': 6})

# plotting histogramms with 5 years interval each
plt.subplot(3, 3, 8)
split = np.split(numbers_series, 5)
plt.hist([split[i] for i in range(0, 5)], bins=8, edgecolor='#2F8525', linewidth=0.5,
         label=['1997-2001', '2002-2006', '2007-2011', '2012-2016', '2017-2021'])
plt.grid(linestyle=':')
plt.legend(loc='upper right', prop={'size': 6})
plt.xlabel("Price dollars per million BTU")
plt.ylabel("Frequency")

# plotting boxplots with 5 years interval each
plt.subplot(3, 3, 9)
bp = plt.boxplot([split[i] for i in range(0, 5)], patch_artist=True)
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=1)
for flier in bp['fliers']:
    flier.set(marker='x', color='#2F8525', alpha=0.5)
plt.grid(linestyle=':')
plt.legend(loc='upper right', prop={'size': 6})
plt.ylabel("Price dollars per million BTU")
plt.xlabel("Time intervals")
plt.show()
