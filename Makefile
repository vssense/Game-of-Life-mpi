bin = bin

$(bin)/main: main.cpp
	mkdir -p $(bin)
	mpicxx -o $(bin)/main main.cpp -std=c++20

$(bin)/main2: main2.cpp
	mkdir -p $(bin)
	mpicxx -o $(bin)/main2 main2.cpp

run: $(bin)/main
	mpirun -n 4 ./$(bin)/main


run2: $(bin)/main2
	mpirun -n 4 ./$(bin)/main2

clean:
	rm -fr $(bin)
