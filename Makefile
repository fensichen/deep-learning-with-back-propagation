FLAGS=-Iinclude/ -g -std=c++11 `PKG_CONFIG_PATH=/home/fensi/local/opencv3.2/lib/pkgconfig pkg-config --cflags opencv` 
all:
	g++ ${FLAGS} -c src/main.cpp
	g++ ${FLAGS} -c src/data.cpp
	g++ ${FLAGS} -c src/relu.cpp
	g++ ${FLAGS} -c src/tanh.cpp
	g++ ${FLAGS} -c src/sigmoid.cpp
	g++ ${FLAGS} -c src/crossentropy.cpp
	g++ ${FLAGS} -c src/gaussian.cpp
	g++ ${FLAGS} -c src/uniform.cpp
	g++ ${FLAGS} -c src/fcn.cpp
	g++ ${FLAGS} -c src/accuracy.cpp
	g++ ${FLAGS} -c src/dropout.cpp
	g++ ${FLAGS} -c src/convolution.cpp
	g++ ${FLAGS} -c src/pooling.cpp
	g++ ${FLAGS} -c src/saver.cpp
	g++ -o dnn *.o `PKG_CONFIG_PATH=/home/fensi/local/opencv3.2/lib/pkgconfig pkg-config --libs opencv` -lboost_system -lboost_filesystem
test:
	g++ ${FLAGS} -c src/test.cpp
	g++ ${FLAGS} -c src/data.cpp
	g++ ${FLAGS} -c src/gaussian.cpp
	g++ ${FLAGS} -c src/convolution.cpp
	g++ ${FLAGS} -c src/pooling.cpp
	g++ -o test test.o data.o gaussian.o convolution.o pooling.o `PKG_CONFIG_PATH=/home/fensi/local/opencv3.2/lib/pkgconfig pkg-config --libs opencv`
clean:
	rm *.o
	rm test
