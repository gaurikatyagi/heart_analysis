import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, detrend

measures = {}


# Preprocessing
def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


def get_samplerate(dataset):
    sampletimer = [x for x in dataset.datetime]
    measures['fs'] = ((len(sampletimer) / sampletimer[-1]) * 1000)


def rolmean(dataset, hrw, fs):
    # mov_avg = pd.rolling_mean(dataset.hart, window=(hrw * fs), center=False)
    mov_avg = dataset.hart.rolling(window=(hrw * fs), center=False).mean()
    avg_hr = (np.mean(dataset.hart))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    dataset['hart_rollingmean'] = mov_avg


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filtersignal(data, cutoff, fs, order):
    hart = [math.pow(x, 3) for x in data.hart]
    hartfiltered = butter_lowpass_filter(hart, cutoff, fs, order)
    data['hart'] = hartfiltered


# Peak detection
def detect_peaks(dataset, ma_perc, fs):
    rolmean = [(x + ((x / 100) * ma_perc)) for x in dataset.hart_rollingmean]
    window = []
    peaklist = []
    listpos = 0
    for datapoint in dataset.hart:
        rollingmean = rolmean[listpos]
        if (datapoint <= rollingmean) and (len(window) <= 1):  # Here is the update in (datapoint <= rollingmean)
            listpos += 1
        elif (datapoint > rollingmean):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(maximum))
            peaklist.append(beatposition)
            window = []
            listpos += 1
    measures['peaklist'] = peaklist
    measures['ybeat'] = [dataset.hart[x] for x in peaklist]
    measures['rolmean'] = rolmean
    calc_RR(dataset, fs)
    measures['rrsd'] = np.std(measures['RR_list'])


def fit_peaks(dataset, fs):
    ma_perc_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200,
                    300]  # List with moving average raise percentages, make as detailed as you like but keep an eye on speed
    rrsd = []
    valid_ma = []
    for x in ma_perc_list:
        detect_peaks(dataset, x, fs)
        bpm = ((len(measures['peaklist']) / (len(dataset.hart) / fs)) * 60)
        rrsd.append([measures['rrsd'], bpm, x])
    for x, y, z in rrsd:
        if ((x > 1) and ((y > 30) and (y < 130))):
            valid_ma.append([x, z])
    measures['best'] = min(valid_ma, key=lambda t: t[0])[1]
    detect_peaks(dataset, min(valid_ma, key=lambda t: t[0])[1], fs)


def check_peaks(dataset):
    RR_list = measures['RR_list']
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    upper_threshold = np.mean(RR_list) + 300
    lower_threshold = np.mean(RR_list) - 300
    removed_beats = []
    removed_beats_y = []
    RR_list_cor = []
    peaklist_cor = [peaklist[0]]
    cnt = 0
    while cnt < len(RR_list):
        if (RR_list[cnt] < upper_threshold) and (RR_list[cnt] > lower_threshold):
            RR_list_cor.append(RR_list[cnt])
            peaklist_cor.append(peaklist[cnt + 1])
            cnt += 1
        else:
            removed_beats.append(peaklist[cnt + 1])
            removed_beats_y.append(ybeat[cnt + 1])
            cnt += 1
    measures['RR_list_cor'] = RR_list_cor
    measures['peaklist_cor'] = peaklist_cor


# Calculating all measures
def calc_RR(dataset, fs):
    peaklist = measures['peaklist']
    RR_list = []
    cnt = 0
    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist)
        cnt += 1
    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list) - 1)):
        RR_diff.append(abs(RR_list[cnt] - RR_list[cnt + 1]))
        RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt + 1], 2))
        cnt += 1
    measures['RR_list'] = RR_list
    measures['RR_diff'] = RR_diff
    measures['RR_sqdiff'] = RR_sqdiff


def calc_ts_measures(dataset):
    RR_list = measures['RR_list_cor']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']
    measures['bpm'] = 60000 / np.mean(RR_list)
    measures['ibi'] = np.mean(RR_list)
    measures['sdnn'] = np.std(RR_list)
    measures['sdsd'] = np.std(RR_diff)
    measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if (x > 20)]
    NN50 = [x for x in RR_diff if (x > 50)]
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))


def calc_fd_measures(dataset, fs):
    peaklist = measures['peaklist_cor']
    RR_list = measures['RR_list_cor']
    RR_x = peaklist[1:]
    RR_y = RR_list
    RR_x_new = np.linspace(RR_x[0], RR_x[-1], len(RR_x))
    f = interp1d(RR_x, RR_y, kind='cubic')
    n = len(dataset.hart)
    frq = np.fft.fftfreq(len(dataset.hart), d=((1 / fs)))
    frq = frq[range(n / 2)]
    Y = np.fft.fft(f(RR_x_new)) / n
    Y = Y[range(n / 2)]
    measures['lf'] = np.trapz(abs(Y[(frq >= 0.04) & (frq <= 0.15)]))
    measures['hf'] = np.trapz(abs(Y[(frq >= 0.16) & (frq <= 0.5)]))


# Plotting it
def plotter(dataset):
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    plt.title("Best fit: mov_avg %s percent raised" % measures['best'])
    plt.plot(dataset.hart, alpha=0.5, color='blue', label="heart rate signal")
    plt.plot(measures['rolmean'], color='green', label="moving average")
    plt.scatter(peaklist, ybeat,
                color='red',
                label="RRSD:%.2f\nBPM:%.2f"%(np.std(measures["RR_list"]), measures['bpm']))
    # , label="average: %.1f BPM" %measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()


# Wrapper function
def process(dataset, hrw, fs):
    filtersignal(dataset, 2.5, fs, 5)
    rolmean(dataset, hrw, fs)
    fit_peaks(dataset, fs)
    calc_RR(dataset, fs)
    check_peaks(dataset)
    calc_ts_measures(dataset)
    # calc_fd_measures(dataset, fs)

if __name__ == "__main__":
    dataset = get_data(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","noisy_data.csv"))
    process(dataset, 0.75, 100)

    # Dictionary "measures" contains all calculated values
    print measures["peaklist_cor"]