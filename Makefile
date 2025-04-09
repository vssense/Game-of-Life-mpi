bin = bin

$(bin)/main: main.cpp
	mkdir -p $(bin)
	mpicxx -o $(bin)/main main.cpp -std=c++20

run: $(bin)/main
	mpirun -n 4 ./$(bin)/main

clean:
	rm -fr $(bin)
