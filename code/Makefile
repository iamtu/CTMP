CC = g++ -Wall
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE = /usr/local/include/gsl/include
GSL_LIB = /usr/local/include/gsl/lib

GSL_INCLUDE_MAC = /usr/local/include/gsl/include/
GSL_LIB_MAC = /usr/local/include/gsl/lib


LSOURCE = main.cpp utils.cpp corpus.cpp ldap.cpp data.cpp
LHEADER = utils.h corpus.h ldap.h data.h

mac: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE_MAC) -L$(GSL_LIB_MAC) $(LSOURCE) -o ldap $(LDFLAGS) 

mac-d: $(LSOURCE) $(HEADER)
	  $(CC) -g -I$(GSL_INCLUDE_MAC) -L$(GSL_LIB_MAC) $(LSOURCE) -o ldap $@ $(LDFLAGS)

linux: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o ldap-condor $(LDFLAGS)

linux-d: $(LSOURCE) $(HEADER)
	  $(CC) -g -I$(GSL_INCLUDE) -L$(GSL_LIB_MAC) $(LSOURCE) -o ldap-condor $@ $(LDFLAGS)

clean:
	-rm -f ldap
clean-d:
	-rm -f ldap
