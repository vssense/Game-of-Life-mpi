main: main.cpp
	mpicxx -o main main.cpp -std=c++20

run: main
	mpirun -n 4 ./main

clean:
	rm -f main