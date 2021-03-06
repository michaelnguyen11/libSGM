all: fast_build

clean:
	rm -rf build

fast_build: clean
	mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4

debug: clean
	mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j4

