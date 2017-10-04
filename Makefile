COMPILER = g++
CFLAGS = -Wall -O2 -lgtest_main -lgtest -lpthread
CXXFLAGS = -std=c++11

GTESTDIR = ~/googletest-master/
GTEST_INCLUDEDIR = $(GTEST_DIR)/googletest/include
GTEST_LIBS = $(GTEST_DIR)/build/googlemock/gtest

SRCDIR = $(shell pwd)/src

test: util_test.cpp read_data.cpp
	$(COMPILER) $(CXXFLAGS) -o test util_test.cpp read_data.cpp -I$(GTEST_INCLUDEDIR) -I$(SRCDIR) -L$(GTEST_LIBDIR) $(CFLAGS)

clean:
	rm -f test util_test.o read_data.o
