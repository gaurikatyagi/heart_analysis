##We will see how to deal with noise in a few stages:
#Evaluate the result of passing this signal to our algorithm from part two;
#Careful: Sampling Frequency;
#Filter the signal to remove unwanted frequencies (noise);
#Improving detection accuracy with a dynamic threshold;
#Detecting incorrectly detected / missed peaks;
#Removing errors and reconstructing the R-R signal to be error-free.

##HARD CODING OF GRAPHS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d
from scipy import signal
import sys

signal_measures = {}
time_measures = {}
frequency_measure = {}

def read_data(filename):
    """
    This function reads in the csv file as a pandas dataframe
    :param filename: String variable which contains the file name to be read
    :return: returns pandas dataframe containing the csv file
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename)
    print "Reading file: %s"%filename
    try:
        data = pd.read_csv(file_path) # We can not make index_col= 0 because the function detect_peaks() makes use of
        #default indices
        print "File read successful"
    except:
        print "File does not exist. Exiting..."
        sys.exit()
    return data

def calc_freq_rate(data):
    sample_timer = [x for x in data["timer"]] # dataset.timer is a ms counter with start of recording at '0'
    frequency_measure["frequency"] = (len(sample_timer)/sample_timer[-1])*100 #Divide total length of dataset by last timer
    # entry. This is in deci second, so multiply by 100 to get Hz value
    return frequency_measure["frequency"]

def rolling_mean(data, window_size, frequency):
    """
    This function calculates a rolling mean signal which averages the given signal over window sizes defined by the
    multiple of window_size and frequency
    :param data: pandas dataframe which stores the dataset
    :param window_size: float value which defines the size of the window to be taken when calculating rolling mean
    :param frequency: frequency of sampling the heart signal
    :return: this function returns the pandas dataframe with an added column for rolling mean values
    """
    moving_average = data["hart"].rolling(window=(window_size*frequency),center=False).mean()
    heart_rate_average = np.mean(data["hart"])
    moving_average = [heart_rate_average if math.isnan(average) else average for average in moving_average]
    data["hart_rolling_mean"] = moving_average
    return data

def butter_lowpass(cutoff, frequency, order = 5): #5th order butterpassfilter
    nyquist_frequency = 0.5*frequency #Nyquist frequency is half the sampling frequency
    normal_cutoff = cutoff/nyquist_frequency
    b, a = signal.butter(order, Wn = normal_cutoff, btype = "low", analog = False)
    #A scalar or length-2 sequence giving the critical frequencies. For a Butterworth filter, this is the point at
    # which the gain drops to 1/sqrt(2) that of the passband
    return b, a

def butter_lowpass_filter(data, cutoff, frequency, order = 5):
    b, a = butter_lowpass(cutoff, frequency, order) #b, a are the filter coefficients
    y = signal.lfilter(b, a, data)
    return y

def fit_peaks(data, fs):
    moving_average_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] #list with
    #moving average raise perccentages
    rr_standard_deviation = []
    for ma in moving_average_list: #detect peaks with all moving average percentages
        detect_peaks(data, ma/100.0)
        R_R_measures(fs)
        bpm = (len(signal_measures["R_positions"])/(len(data["hart"])/fs)*60)
        rr_standard_deviation.append((signal_measures["RR_standard_deviation"], bpm, ma))
    for sd, bpm_item, ma_item in rr_standard_deviation:
        if (sd>1 and (bpm_item>30 and bpm_item<130)):
            signal_measures["best"]= [sd, ma_item] #the items in rr_standard_deviation are sorted by moving average as they
            #are in the same sequence we fed in
            # print signal_measures["best"]
            break
    # detect_peaks(data, signal_measures["best"][1], fs)
    detect_peaks(data, (signal_measures["best"][1]/100.0))
    R_R_measures(fs)

def detect_peaks(data, moving_average_percent):
    """
    This function detects the peak values i.e the R values in the signal of the type QRS complex. It adds to the
    measures dictionary, all the R values and their x coordinates
    :param data: pandas dataframe which stores the dataset
    :param moving_average_percent: integer value which stores the trial value of moving average
    :param freq: calculated frequency of the signal
    :return: This function does not return anything. It updates the dictionary, signal_measures
    """
    # print moving_average_percent
    roll_mean_threshold = [(x+(x*moving_average_percent)) for x in data["hart_rolling_mean"]]
    # print roll_mean_threshold
    window = []
    #window is a temporary variable which stores all the values above rolling mean till a signal again does not drop
    # equal to or lower than the mean. From this list, we find the highest value and call it R in the QRS complex
    peak_position_list = []
    peak_values = []
    mean_for_R = None
    for position, datapoint in enumerate(data["hart"]):
        rolling_mean_value = roll_mean_threshold[position]
        if (datapoint>rolling_mean_value):
            # window.append(datapoint)
            if (position>100 and rolling_mean_value>=np.mean(roll_mean_threshold[:position])):
                window.append(datapoint)
            elif (position<=100):
                window.append(datapoint)
            else:
                pass
        elif (datapoint<=rolling_mean_value and len(window)<=1):
            #When dta point is less than the rolling mean and the window hasn't started yet, do nothing
            pass
        else:
            #all other cases, datapoint is less than the mean but there were vaues in the window, that means it is time
            #to detect R values.
            R_value = max(window)
            R_position = position-len(window)+window.index(R_value) #current position-length of window gets us to the
            #starting of the window. Then we add the index value of the identified R so as to add it to the initial
            # window value to get the position of R peak.
            peak_position_list.append(R_position)
            peak_values.append(R_value)
            window = [] #Re-initialize window and repeat
    signal_measures["R_positions"] = peak_position_list
    signal_measures["R_values"] = peak_values
    signal_measures["roll_mean"] = roll_mean_threshold

def R_R_measures(frequency):
    """
    This function finds the x-coordinate distance between each consecutive R value, its square and the distance in terms
    of miliseconds
    :param data: pandas dataframe which stores the dataset
    :param frequency: integer value which gives the sampling frequency of heart signal
    :return: This function does not return anything. It updates the dictionary, signal_measures
    """
    R_positions = signal_measures["R_positions"]
    RR_msdiff = [] #Stores the mili second distance between consecutive R values
    for position in range(len(R_positions)-1):
        RR_interval = R_positions[position+1]-R_positions[position]
        distance_ms = (RR_interval/frequency)*1000.0
        RR_msdiff.append(distance_ms)
    RR_diff = []
    RR_sqdiff = []
    for position in range(len(R_positions) - 1):
        RR_diff.append(abs(R_positions[position] - R_positions[position+1]))
        RR_sqdiff.append(math.pow(abs(R_positions[position] - R_positions[position+1]), 2))
    signal_measures["RR_msdiff"] = RR_msdiff
    signal_measures["RR_diff"] = RR_diff # difference in positions of consecutive R values in terms of x values
    signal_measures["RR_sqdiff"] = RR_sqdiff
    signal_measures["RR_standard_deviation"] = np.std(signal_measures["RR_msdiff"])

def calc_frequency_measures(data, frequency):
    R_positions = signal_measures["R_positions"][1:]
    RR_msdiff = signal_measures["RR_msdiff"]
    RR_msdiff_x = np.linspace(R_positions[0], R_positions[-1], R_positions[-1])
    func = interp1d(R_positions, RR_msdiff, kind = "cubic") #interpolation to finf the possible x values which might
    # have been missed and to later deal with them as they will impact the frequency
    n = len(data["hart"])
    frequency_f = np.fft.fftfreq(n = n, d = (1.0/frequency)) #Return the Discrete Fourier Transform sample frequencies.
    #Given a window length n and a sample spacing d. The returned float array contains the frequency bin centers in
    # cycles per unit of the sample spacing (with zero at the start). For instance, if the sample spacing is in seconds,
    # then the frequency unit is cycles/second.
    frequency_f = frequency_f[range(n/2)] #taking the upper half values of frequencies which have been calculated by fft
    y = np.fft.fft(func(RR_msdiff_x))/n #Compute the one-dimensional n-point discrete Fourier Transform for equally
    # spaced x values
    y = y[range(n/2)] #taking the upper half values
    frequency_measure["lf"] = np.trapz(abs(y[(frequency_f>=0.04) & (frequency_f<=0.15)]))
    frequency_measure["hf"] = np.trapz(abs(y[(frequency_f>=0.16) & (frequency_f<=0.5)]))


if __name__ == "__main__":
    data = read_data("noisy_data.csv")
    # print data
    frequency = calc_freq_rate(data)
    print "Frequency of the data is: ", frequency
    # print data["hart"]
    filtered = butter_lowpass_filter(data["hart"], 2.5, frequency, 5)

    file_save_path = os.path.dirname(os.path.abspath(__file__))

    plt.subplot(211)
    plt.plot(data["hart"], color = "red", label = "original hart", alpha = 0.5)
    plt.legend(loc = "best")
    plt.subplot(212)
    plt.plot(filtered, color="green", label="filtered hart", alpha=0.5)
    plt.legend(loc = "best")
    plt.suptitle("original v/s filtered data")
    plt.savefig(os.path.join(file_save_path, "tmp","orig_cleaned.jpeg"))
    data["hart"] = filtered

    roll_mean_data = rolling_mean(data, window_size=0.75, frequency=frequency)
    # print roll_mean_data
    fit_peaks(roll_mean_data, frequency)
    calc_frequency_measures(roll_mean_data, frequency)
    time_measures["bpm"] = (len(signal_measures["R_positions"]) / (len(roll_mean_data["hart"]) / frequency) * 60)


    ## Whenever signal drops below the signal threshold we get it to the threshold value
    # new_signal = [signal_measures["roll_mean"][i] if roll_mean_data["hart"][i]< signal_measures["roll_mean"][i]
    #           else roll_mean_data["hart"][i] for i in range(len(roll_mean_data["hart"]))]
    #
    # roll_mean_data["hart"] = new_signal

    #plot R, MA and signal- All peaks are R-peaks
    R_positions = signal_measures["R_positions"]
    ybeat = signal_measures["R_values"]
    # print roll_mean_data["hart_rolling_mean"]
    # print ybeat
    plt.subplot(111)
    plt.title("Heart Rate signal with moving average")
    plt.plot(roll_mean_data["hart"], alpha=0.5, color='blue', label="raw signal")
    plt.plot(roll_mean_data["hart_rolling_mean"], color='green', label="moving average")
    plt.scatter(R_positions, ybeat, color='red', label="average: %.1f BPM" % time_measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.savefig(os.path.join(file_save_path, "tmp","Rs_identified.jpeg"))

    # print signal_measures["best"]
    # print signal_measures["roll_mean"][100]
    # print roll_mean_data["hart_rolling_mean"][100]
    # print all(roll_mean_data["hart_rolling_mean"] == signal_measures["roll_mean"])