COMPILER = g++
CFLAGS = -Wall -O2
TESTFLAGS = -lgtest_main -lgtest -lpthread
CXXFLAGS = -std=c++11

BINDIR := bin

GTESTDIR = $(shell echo "$(HOME)")/googletest-master
GTEST_INCLUDEDIR = $(GTESTDIR)/googletest/include
GTEST_LIBS = $(GTESTDIR)/build/googlemock/gtest

SRCDIR = $(shell pwd)/src
TESTDIR = $(shell pwd)/test

.PHONY: all
all: $(BINDIR)/mlp $(BINDIR)/cnn $(BINDIR)/utest

$(BINDIR)/mlp: src/mlp/mlp_main.cpp src/util/read_data.cpp $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ src/mlp/mlp_main.cpp src/util/read_data.cpp -I$(SRCDIR) $(CFLAGS)

$(BINDIR)/cnn: src/cnn/cnn_main.cpp src/util/read_data.cpp $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ src/cnn/cnn_main.cpp src/util/read_data.cpp -I$(SRCDIR) $(CFLAGS)

$(BINDIR)/utest: test/util_test.cpp src/util/read_data.cpp $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ test/util_test.cpp src/util/read_data.cpp -I$(GTEST_INCLUDEDIR) -I$(SRCDIR) -L$(GTEST_LIBDIR) $(CFLAGS) $(TESTFLAGS)

.PHONY: clean
clean:
	rm -rf $(BINDIR)/utest $(BINDIR)/mlp $(BINDIR)/cnn test/util_test.o src/util/read_data.o src/mlp/mlp_main.o src/cnn/cnn_main.o $(BINDIR)

$(BINDIR):
	mkdir -p $(BINDIR)
