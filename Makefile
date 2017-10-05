COMPILER = g++
CFLAGS = -Wall -O2
TESTFLAGS = -lgtest_main -lgtest -lpthread
CXXFLAGS = -std=c++11

GTESTDIR = $(shell echo "$(HOME)")/googletest-master
GTEST_INCLUDEDIR = $(GTESTDIR)/googletest/include
GTEST_LIBS = $(GTESTDIR)/build/googlemock/gtest

SRCDIR = $(shell pwd)/src
TESTDIR = $(shell pwd)/test

mlp: src/mlp/mlp_main.cpp src/util/read_data.cpp
	$(COMPILER) $(CXXFLAGS) -o mlp src/mlp/mlp_main.cpp src/util/read_data.cpp -I$(SRCDIR) $(CFLAGS)

test: test/util_test.cpp src/util/read_data.cpp
	$(COMPILER) $(CXXFLAGS) -o utiltest test/util_test.cpp src/util/read_data.cpp -I$(GTEST_INCLUDEDIR) -I$(SRCDIR) -L$(GTEST_LIBDIR) $(CFLAGS) $(TESTFLAGS)

clean:
	rm -f utiltest test/util_test.o src/util/ead_data.o

