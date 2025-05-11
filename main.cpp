#include <mpi.h>

#include <cassert>
#include <span>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>

const int kGlobalWidth = 8;
const int kGlobalHeight = 8;
const int kMaxGenerations = 40;

enum Side
{
    kUp = 0,
    kDown = 1,
    kLeft = 2,
    kRight = 3
};

/// + - + -
/// - + + -
/// - + - -
/// - - - -
const std::vector<std::pair<int, int>> kStartingPosition = {
    {0, 0},
    {0, 2},
    {1, 1},
    {1, 2},
    {2, 1},

    // {0, 0},
    // {1, 1},
    // {2, 2},
    // {3, 3},
    // {4, 4},
    // {5, 5},
    // {6, 6},
    // {7, 7},
    // {8, 8},
};

class Grid
    : public std::vector<int>
{
public:
    Grid() = default;
    Grid(int rows, int cols)
        : std::vector<int>(rows * cols, 0)
        , m_rows(rows)
        , m_cols(cols)
    {
        assert(rows > 0);
        assert(cols > 0);
    }

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }

    auto operator[](int index)
    {
        assert(index < rows());
        return std::span<int>(begin() + index * m_cols, m_cols);
    }

    auto operator[](int index) const
    {
        assert(index < rows());
        return std::span<const int>(cbegin() + index * m_cols, m_cols);
    }

private:
    int m_rows = -1;
    int m_cols = -1;
};

void print_grid(const Grid& grid)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss;

    ss << "---------------[" << rank << "]------------------\n";
    for (int row = 0; row < grid.rows(); ++row)
    {
        for (int alive : grid[row])
        {
            ss << (alive ? " + " : " - ");
        }
        ss << '\n';
    }

    ss << "------------------------------------";

    std::cout << ss.str() << std::endl;
}

void scatter_grid(const Grid& global, Grid& subdom, MPI_Datatype block_t)
{
    MPI_Scatter(global.data(), 1, block_t, subdom[1].data(), 1, block_t, 0, MPI_COMM_WORLD);
}

void gather_grid(Grid& global, const Grid& subdom, MPI_Datatype block_t)
{
    MPI_Gather(subdom[1].data(), 1, block_t, global.data(), 1, block_t, 0, MPI_COMM_WORLD);
}

void sync_borders(Grid& subdom, MPI_Datatype row_t)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size; 
    int prev = rank == 0 ? size - 1 : rank - 1;

    MPI_Request reqs[4];
    MPI_Status stats[4];

    MPI_Isend(subdom[1].data(), 1, row_t, prev, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(subdom[0].data(), 1, row_t, prev, 1, MPI_COMM_WORLD, &reqs[1]);

    int rows = subdom.rows() - 2;

    MPI_Isend(subdom[rows    ].data(), 1, row_t, next, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(subdom[rows + 1].data(), 1, row_t, next, 0, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, stats);
}

size_t grid_hash(const Grid& grid)
{
    size_t hash = grid.size();
    for (const auto& i : grid)
    {
        hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    return hash;
}
  
bool test_stop(const Grid& subdom)
{
    static std::unordered_set<size_t> hashes;

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t subdom_hash = grid_hash(subdom);
    int new_subdom = (hashes.find(subdom_hash) == hashes.end());
    hashes.insert(subdom_hash);

    std::vector<int> new_subdoms(size, 0);
    MPI_Allgather(&new_subdom, 1, MPI_INT, (int*)new_subdoms.data(), 1, MPI_INT, MPI_COMM_WORLD);

    return std::find(new_subdoms.begin(), new_subdoms.end(), 1) == new_subdoms.end(); // find new subdom
}

int divide_for_rank(int val, int rank, int size)
{
    if (rank != size - 1)
    {
        return val / size;
    }
    else
    {
        return val / size + val % size;
    }
}

int get_subdom_height(int rank, int size)
{
    return divide_for_rank(kGlobalHeight, rank, size);
}

int get_subdom_width(int rank, int size)
{
    return divide_for_rank(kGlobalWidth, rank, size);
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int subdom_width = kGlobalWidth;
    int subdom_height = get_subdom_height(rank, size);
    
    MPI_Datatype row_t;
    MPI_Type_contiguous(subdom_width, MPI_INT, &row_t);
    MPI_Type_commit(&row_t);
    
    MPI_Datatype block_t;
    MPI_Type_contiguous(subdom_width * subdom_height, MPI_INT, &block_t);
    MPI_Type_commit(&block_t);

    Grid subdom(subdom_height + 2, subdom_width);
    Grid global;

    if (rank == 0)
    {
        global = Grid(kGlobalHeight, kGlobalWidth);
        for (const auto& alive_coords : kStartingPosition)
        {
            global[alive_coords.first][alive_coords.second] = 1;
        }

        print_grid(global);
    }

    scatter_grid(global, subdom, block_t);

    sync_borders(subdom, row_t);
    test_stop(subdom); // save gen 0 position

    // print_grid(subdom);
    // return 0;

    int gen = 0;
    for (; gen < kMaxGenerations; ++gen)
    {
        Grid new_subdom = subdom;
        for (int i = 1; i < subdom.rows() - 1; ++i)
        {
            for (int j = 0; j < subdom.cols(); ++j)
            {
                int prev_j = j == 0 ? subdom.cols() - 1 : j - 1;
                int next_j = j == subdom.cols() - 1 ? 0 : j + 1;

                int neighbors_count =
                    subdom[i - 1][prev_j] + subdom[i - 1][j] + subdom[i - 1][next_j] +
                    subdom[i]    [prev_j] + subdom[i][next_j] +
                    subdom[i + 1][prev_j] + subdom[i + 1][j] + subdom[i + 1][next_j];
                
                new_subdom[i][j] = (neighbors_count == 3) || 
                                   (subdom[i][j] && neighbors_count == 2);
            }
        }

        subdom = std::move(new_subdom);
        sync_borders(subdom, row_t);

        if (test_stop(subdom))
        {
            break;
        }
    }

    gather_grid(global, subdom, block_t);
    if (rank == 0)
    {
        if (gen == kMaxGenerations)
        {
            std::cout << "Max generations reached" << std::endl;
        }
        else
        {
            std::cout << "World repeated itself on gen = " << gen << std::endl;
        }

        print_grid(global);
    }

    MPI_Type_free(&row_t);
    MPI_Type_free(&block_t);

    MPI_Finalize();
    return 0;
}