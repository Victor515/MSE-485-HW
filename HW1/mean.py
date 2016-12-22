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

a = [1,2,3,4]
print mean(a)
