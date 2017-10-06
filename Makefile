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

cnn: src/cnn/cnn_main.cpp src/util/read_data.cpp
	$(COMPILER) $(CXXFLAGS) -o cnn src/cnn/cnn_main.cpp src/util/read_data.cpp -I$(SRCDIR) $(CFLAGS)

utest: test/util_test.cpp src/util/read_data.cpp
	$(COMPILER) $(CXXFLAGS) -o utest test/util_test.cpp src/util/read_data.cpp -I$(GTEST_INCLUDEDIR) -I$(SRCDIR) -L$(GTEST_LIBDIR) $(CFLAGS) $(TESTFLAGS)

clean:
	rm -f utest mlp cnn test/util_test.o src/util/read_data.o src/mlp/mlp_main.o src/cnn/cnn_main.o

