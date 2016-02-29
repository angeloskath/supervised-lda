
CC = g++-4.9
CFLAGS = -std=c++11 -Wall -O3 -msse2
LDFLAGS = -lm -ldocopt
LDFLAGS_TEST = -lgtest

INCLUDE_DIRS = include/

SOURCES = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp) $(wildcard src/*/*/*.cpp)
OBJECTS = $(patsubst src/%.cpp, build/%.o, $(SOURCES))
TEST_SOURCES = $(wildcard test/*.cpp)
TEST_BINARIES = $(patsubst test/%.cpp, bin/%, $(TEST_SOURCES))

all: bin/slda

build/%.o : src/%.cpp include/%.hpp
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@ -I $(INCLUDE_DIRS)

bin/slda: slda.cpp $(OBJECTS)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS) slda.cpp $(LDFLAGS) -I $(INCLUDE_DIRS) -o bin/slda

bin/%: test/%.cpp $(OBJECTS)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS) -I $(INCLUDE_DIRS) $< -o $@ $(LDFLAGS) $(LDFLAGS_TEST)

check: $(TEST_BINARIES)
	$(foreach test, $(TEST_BINARIES), $(test);)

clean:
	rm -rf build
	rm -rf bin
