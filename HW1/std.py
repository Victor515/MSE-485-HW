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
    total  = 0.0
    num    = len(trace)
    average   = mean(trace)

    for i in range(num):
        total += (average - trace[i])**2

    stddev = (total/(num-1))**0.5
    # calculate stadard deviation
    return stddev

# end def std

a = [1.12,1.52,1.33,1.09,1.20,1.26,1.44,1.34,1.19,1.13,1.56,1.45]
print mean(a),std(a)**2
