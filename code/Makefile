CC = g++ -Wall -std=c++11
LDFLAGS = -lgsl -lm -lgslcblas -lpthread

#GSL_INCLUDE_MAC = /usr/local/include
#GSL_LIB_MAC = /usr/local/lib

LSOURCE = main.cpp utils.cpp corpus.cpp ldap.cpp data.cpp
LHEADER = utils.h corpus.h ldap.h data.h

mac: $(LSOURCE) $(HEADER)
	$(CC) $(LSOURCE) -o ldap $(LDFLAGS) 
	#$(CC) -I$(GSL_INCLUDE_MAC) -L$(GSL_LIB_MAC) $(LSOURCE) -o ldap $(LDFLAGS) 
clean:
	-rm -f ldap
