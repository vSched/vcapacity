CC = g++
CFLAGS = -Wall -std=c++17

TARGET = vcap

all: $(TARGET)

$(TARGET): main.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o -O3

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm -f *.o $(TARGET)
