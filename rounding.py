import numpy

ceil  = numpy.ceil
floor = numpy.floor
normf = lambda x : x/abs(x+(x==0))
normh = lambda x : normf(x) + (x==0)
head  = lambda x : ceil(abs(x))*normh(x)
foot  = lambda x : floor(abs(x))*normf(x)
