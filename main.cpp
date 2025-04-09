#include <mpi.h>

#include <cassert>
#include <span>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>

const int kGlobalWidth = 9;
const int kGlobalHeight = 9;
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

std::pair<std::vector<int>, std::vector<int>> GetDisplacementsAndRecvCounts(int size)
{
    std::vector<int> recvcounts(size, kGlobalWidth * (kGlobalHeight / size));
    recvcounts.back() = kGlobalWidth * (kGlobalHeight / size + kGlobalHeight % size);

    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i)
    {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    return { std::move(displs), std::move(recvcounts) };
}

void scatter_grid(const Grid& global, Grid& subdom)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto& [displs, recvcounts] = GetDisplacementsAndRecvCounts(size);

    assert(rank != 0 || global.size() == recvcounts.back() + displs.back());

    MPI_Scatterv(global.data(), recvcounts.data(), displs.data(), MPI_INT, (int*)subdom[1].data(), recvcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);
}

void gather_grid(const Grid& global, Grid& subdom)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const auto& [displs, recvcounts] = GetDisplacementsAndRecvCounts(size);
    
    assert(rank != 0 || global.size() == recvcounts.back() + displs.back());

    MPI_Gatherv(subdom[1].data(), recvcounts[rank], MPI_INT, (int*)global.data(), recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
}

void sync_borders(Grid& subdom)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size; 
    int prev = rank == 0 ? size - 1 : rank - 1;

    MPI_Request reqs[4];
    MPI_Status stats[4];

    MPI_Isend(subdom[1].data(), subdom[1].size(), MPI_INT, prev, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv((int*)subdom[0].data(), subdom[0].size(), MPI_INT, prev, 1, MPI_COMM_WORLD, &reqs[1]);

    int rows = subdom.rows() - 2;

    MPI_Isend(subdom[rows].data(), subdom[rows].size(), MPI_INT, next, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv((int*)subdom[rows + 1].data(), subdom[rows + 1].size(), MPI_INT, next, 0, MPI_COMM_WORLD, &reqs[3]);

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

int get_subdom_height(int rank, int size)
{
    if (rank != size - 1)
    {
        return kGlobalHeight / size;
    }
    else
    {
        return kGlobalHeight / size + kGlobalHeight % size;
    }
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

    scatter_grid(global, subdom);

    sync_borders(subdom);
    test_stop(subdom); // save gen 0 position

    // print_grid(subdom);

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
        sync_borders(subdom);

        if (test_stop(subdom))
        {
            break;
        }
    }

    gather_grid(global, subdom);
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

    MPI_Finalize();
    return 0;
}