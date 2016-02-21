
CC = g++
CFLAGS = -std=c++11 -O3 -Wall -g
LDFLAGS = -lm

INCLUDE_DIRS = include/

SOURCES = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp) $(wildcard src/*/*/*.cpp)
OBJECTS = $(patsubst src/%.cpp, build/%.o, $(SOURCES))

all: slda

build/%.o : src/%.cpp build_dir
	$(CC) $(CFLAGS) -c $< -o $@ -I $(INCLUDE_DIRS)

slda: $(OBJECTS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) slda.cpp -o bin/slda

build_dir:
	mkdir -p build

clean:
	rm -rf build
	rm -rf bin
