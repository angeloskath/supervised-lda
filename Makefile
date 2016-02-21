
CC = g++
CFLAGS = -std=c++11 -Wall -g -O3
LDFLAGS = -lm

INCLUDE_DIRS = include/

SOURCES = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp) $(wildcard src/*/*/*.cpp)
OBJECTS = $(patsubst src/%.cpp, build/%.o, $(SOURCES))

all: bin/slda

build/%.o : src/%.cpp build
	$(CC) $(CFLAGS) -c $< -o $@ -I $(INCLUDE_DIRS)

bin/slda: $(OBJECTS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) -I $(INCLUDE_DIRS) slda.cpp -o bin/slda

build:
	mkdir -p build

clean:
	rm -rf build
	rm -rf bin
