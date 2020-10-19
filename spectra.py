def Spectra(data,tau,plot = False,filter_order = 1,filter_size = 11):
    # Spectra: generates spectra from a set of data and filters it using a smoothing filter
    # Inputs:  data - number of variables by time numpy array
    #          tau - time step between data points
    #          plot - boolean for plotting
    #          filter_order - order of polynomial smoothing
    #          filter_size - size of filter window
    # Outputs: filtered_spec - one-sided spectrum for each variable
    #          freq - vector of frequency values
    from scipy.signal import periodogram
    from scipy.signal import savgol_filter
    
    freq, spec = periodogram(data, fs = 1./tau, window = 'hamming')
    filtered_spec = savgol_filter(spec, filter_size, filter_order)
    
    if plot:
        PlotSpectrum(filtered_spec,freq)
    
    return filtered_spec, freq

def PlotSpectra(filtered_spec, freq):
    # PlotSpectra: plots the spectra
    
    if len(filtered_spec.shape) == 2:
        for var in range(filtered_spec.shape[0]):
            plt.plot(freq, filtered_spec[var])
            plt.xlabel('Frequency (1/t)')
            plt.ylabel('Filtered Spectral Density')
            plt.title('Variable %d' % var)
            plt.show()
    else:
        plt.plot(freq, filtered_spec)
        plt.xlabel('Frequency (1/t)')
        plt.ylabel('Filtered Spectral Density')
        plt.show()
        
def CompareSpectra(filtered_spec, filtered_true_spec, freq):
    # CompareSpectra: Compares the true spectra with that predicted from the reservoir
    
    if len(filtered_spec.shape) == 2:
        for var in range(filtered_spec.shape[0]):
            plt.plot(freq, filtered_true_spec[var],label = 'Truth')
            plt.plot(freq, filtered_spec[var],label = 'Reservoir')
            plt.xlabel('Frequency (1/t)')
            plt.ylabel('Filtered Spectral Density')
            plt.title('Variable %d' % var)
            plt.legend()
            plt.show()
    else:
        plt.plot(freq, filtered_true_spec,label = 'Truth')
        plt.plot(freq, filtered_spec,label = 'Reservoir')
        plt.xlabel('Frequency (1/t)')
        plt.ylabel('Filtered Spectral Density')
        plt.legend()
        plt.show()
