def mean(trace):
    """ calculate the mean of a trace of scalar data
    results should be identical to np.mean(trace)
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the mean of this trace of scalars 
    """

    total = 0.0        # store sum of all data points
    num   = len(trace) # count total number of data points

    for i in range(num):
        total += trace[i]
    # end for i

    return total/float(num)

# end def mean

def std(trace):
    """ calculate the standard deviation of a trace of scalar data
    results should be identical to np.std(trace)
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the standard deviation of this trace of scalars 
    """

    stddev = 0.0
    num    = len(trace)
    average   = mean(trace) #calculate the average of trace
    total  =0.0

    # calculate stadard deviation
    for i in range(num):
        total += (average - trace[i])**2

    stddev = (total/(num-1))**0.5
    return stddev

# end def std

def corr(trace):
    """ calculate the autocorrelation of a trace of scalar data
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the autocorrelation of this trace of scalars 
    """

    correlation = 1.0
    num              = len(trace)
    average          = mean(trace)
    mystd            = std(trace)
    total            = 0.0
    # calculate auto correlation
    for i in range(num):
        for t in range(num-i):
            total = (trace[t]-average)*(trace[t+i]-average)
        total = total/((num-i)*(mystd**2))
        #end for num-i
        
        if total > 0:
            correlation += 2*total
        else:
            break
        #if correlation<=0, for loop ends(definition of cutoff)
            
    
    return correlation

# end def corr

a = [1,2,3,4,5,6,100,101]
print corr(a)
