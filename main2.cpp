#include <mpi.h>

#include <iostream>

int main()
{
    MPI_Init(NULL, NULL);
    
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const int gsize = 4;
    int g[gsize] = {};
    int l[2] = {-1, 5};

    if (rank == 0)
    {

        for (int i = 0; i < gsize; ++i)
        {
            std::cout << g[i] << ' ';
        }
        std::cout << '\n';
    }

    MPI_Datatype displaced_block_t;
    long lb = -1;
    long extent = -1;

    MPI_Type_get_extent(MPI_INT, &lb, &extent);
    MPI_Type_create_resized(MPI_INT, lb + sizeof(int), extent, &displaced_block_t);
    MPI_Type_commit(&displaced_block_t);
    
    MPI_Gather(l, 1, displaced_block_t, g, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < gsize; ++i)
        {
            std::cout << g[i] << ' ';
        }
        std::cout << '\n';
    }

    MPI_Type_free(&displaced_block_t);


    MPI_Finalize();
}