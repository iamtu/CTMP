CC = g++ -Wall -std=c++11
LDFLAGS = -lgsl -lm -lgslcblas -lpthread

GSL_INCLUDE = /data00/tu.vu/usr/local/include
GSL_LIB = /data00/tu.vu/usr/local/lib

LSOURCE = main.cpp utils.cpp corpus.cpp ldap.cpp data.cpp
LHEADER = utils.h corpus.h ldap.h data.h

linux: $(LSOURCE) $(HEADER)
	$(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o ldap $(LDFLAGS)

mac: $(LSOURCE) $(HEADER)
	$(CC) $(LSOURCE) -o ldap $(LDFLAGS) 

clean:
	-rm -f ldap
