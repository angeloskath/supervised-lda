
CC = g++
CFLAGS = -std=c++11 -Wall -g -O3 -msse2
LDFLAGS = -lm
LDFLAGS_TEST = -lgtest

INCLUDE_DIRS = include/

SOURCES = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp) $(wildcard src/*/*/*.cpp)
OBJECTS = $(patsubst src/%.cpp, build/%.o, $(SOURCES))
TEST_SOURCES = $(wildcard test/*.cpp)
TEST_BINARIES = $(patsubst test/%.cpp, bin/%, $(TEST_SOURCES))

all: bin/slda

build/%.o : src/%.cpp include/%.hpp build
	$(CC) $(CFLAGS) -c $< -o $@ -I $(INCLUDE_DIRS)

bin/slda: $(OBJECTS) bin
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) -I $(INCLUDE_DIRS) slda.cpp -o bin/slda

bin/%: test/%.cpp $(OBJECTS) bin
	$(CC) $(CFLAGS) $(OBJECTS) -I $(INCLUDE_DIRS) $< -o $@ $(LDFLAGS) $(LDFLAGS_TEST)

check: $(TEST_BINARIES)
	$(foreach test, $(TEST_BINARIES), $(test);)

build:
	mkdir -p build

bin:
	mkdir -p bin

clean:
	rm -rf build
	rm -rf bin
