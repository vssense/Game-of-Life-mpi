#include <mpi.h>

#include <cassert>
#include <span>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>

const int kNDim = 2;

const int kGlobalWidth = 8;
const int kGlobalHeight = 8;
const int kMaxGenerations = 40;

enum Side
{
    kTop = 0,
    kBottom,
    kLeft,
    kRight,

    kTopLeft,
    kTopRight,
    kBottomLeft,
    kBottomRight,
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

    // {0, 0},
    // {0, 1},
    // {0, 2},
    // {0, 3},
    // {0, 4},
    // {0, 5},
    // {0, 6},
    // {0, 7},

    // {0, 0},
    // {1, 0},
    // {2, 0},
    // {3, 0},
    // {4, 0},
    // {5, 0},
    // {6, 0},
    // {7, 0},
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
        return std::span<int>(data() + index * m_cols, m_cols);
    }

    auto operator[](int index) const
    {
        assert(index < rows());
        return std::span<const int>(data() + index * m_cols, m_cols);
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

std::pair<std::vector<int>, std::vector<int>> GetDisplacementsAndRecvCounts(int size, const int dims[kNDim], int subdom_width, int subdom_height)
{
    std::vector<int> recvcounts(size, 1);  // Each process receives 1 block
    std::vector<int> displs(size, 0); // in bytes as block_t extent == size == 1
    
    assert(size == dims[0] * dims[1]);

    for (int i = 1; i < size; ++i)
    {
        displs[i] = displs[i - 1] + subdom_width * sizeof(int);
    }

    int ofs = (subdom_height - 1) * kGlobalWidth * sizeof(int); // (subdom_height - 1) because one of heights accounted in previous cycle
    int idx = dims[0];
    for (int i = 1; i < dims[1]; ++i)
    {
        for (int j = 0; j < dims[0]; ++j)
        {
            displs[idx++] += ofs;                           // offset of y coordinate
        }

        ofs += (subdom_height - 1) * kGlobalWidth * sizeof(int);
    }

    assert(idx == size);

    return {std::move(displs), std::move(recvcounts)};
}

void scatter_grid(const Grid& global, Grid& subdom, const int dims[kNDim], MPI_Datatype block_t, MPI_Datatype displaced_block_t)
{
    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto& [displs, recvcounts] = GetDisplacementsAndRecvCounts(size, dims, subdom.cols() - 2, subdom.rows() - 2);

    MPI_Scatterv(global.data(), recvcounts.data(), displs.data(), block_t, &subdom[1][1], 1, displaced_block_t, 0, MPI_COMM_WORLD);
}

void gather_grid(Grid& global, const Grid& subdom, const int dims[kNDim], MPI_Datatype block_t, MPI_Datatype displaced_block_t)
{
    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto& [displs, recvcounts] = GetDisplacementsAndRecvCounts(size, dims, subdom.cols() - 2, subdom.rows() - 2);

    MPI_Gatherv(&subdom[1][1], 1, displaced_block_t, global.data(), recvcounts.data(), displs.data(), block_t, 0, MPI_COMM_WORLD);
}

void sync_borders(Grid& subdom, int neighbors[4], MPI_Datatype col_t, MPI_Datatype row_t, MPI_Comm cart_comm)
{
    int cols = subdom.cols() - 2;
    int rows = subdom.rows() - 2;
    
    MPI_Request reqs[8];
    MPI_Status stats[8];

    MPI_Irecv(&subdom[0][1], 1, row_t, neighbors[kTop], 0, cart_comm, &reqs[1]);
    MPI_Isend(&subdom[1][1], 1, row_t, neighbors[kTop], 1, cart_comm, &reqs[0]);

    MPI_Isend(&subdom[rows    ][1], 1, row_t, neighbors[kBottom], 0, cart_comm, &reqs[2]);
    MPI_Irecv(&subdom[rows + 1][1], 1, row_t, neighbors[kBottom], 1, cart_comm, &reqs[3]);

    MPI_Irecv(&subdom[1][0], 1, col_t, neighbors[kLeft], 2, cart_comm, &reqs[4]);
    MPI_Isend(&subdom[1][1], 1, col_t, neighbors[kLeft], 3, cart_comm, &reqs[5]);

    MPI_Isend(&subdom[1][cols    ], 1, col_t, neighbors[kRight], 2, cart_comm, &reqs[6]);
    MPI_Irecv(&subdom[1][cols + 1], 1, col_t, neighbors[kRight], 3, cart_comm, &reqs[7]);

    MPI_Waitall(8, reqs, stats);

    // corners
    MPI_Irecv(&subdom[0][0], 1, MPI_INT, neighbors[kTopLeft], 0, cart_comm, &reqs[1]);
    MPI_Isend(&subdom[1][1], 1, MPI_INT, neighbors[kTopLeft], 1, cart_comm, &reqs[0]);

    MPI_Isend(&subdom[rows    ][cols    ], 1, MPI_INT, neighbors[kBottomRight], 0, cart_comm, &reqs[6]);
    MPI_Irecv(&subdom[rows + 1][cols + 1], 1, MPI_INT, neighbors[kBottomRight], 1, cart_comm, &reqs[7]);
    
    MPI_Isend(&subdom[rows    ][1], 1, MPI_INT, neighbors[kBottomLeft], 0, cart_comm, &reqs[2]);
    MPI_Irecv(&subdom[rows + 1][0], 1, MPI_INT, neighbors[kBottomLeft], 1, cart_comm, &reqs[3]);
    
    MPI_Irecv(&subdom[0][cols + 1], 1, MPI_INT, neighbors[kTopRight], 0, cart_comm, &reqs[4]);
    MPI_Isend(&subdom[1][cols    ], 1, MPI_INT, neighbors[kTopRight], 1, cart_comm, &reqs[5]);

    MPI_Waitall(8, reqs, stats);
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

    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t subdom_hash = grid_hash(subdom);
    int new_subdom = (hashes.find(subdom_hash) == hashes.end());
    hashes.insert(subdom_hash);

    std::vector<int> new_subdoms(size, 0);
    MPI_Allgather(&new_subdom, 1, MPI_INT, (int*)new_subdoms.data(), 1, MPI_INT, MPI_COMM_WORLD);

    return std::find(new_subdoms.begin(), new_subdoms.end(), 1) == new_subdoms.end(); // find new subdom
}

int GetCartCoords(MPI_Comm cart_comm, const int own_coords[], int dx, int dy)
{
    int rank = -1;
    const int coords[2] = { own_coords[0] + dx, own_coords[1] + dy };

    MPI_Cart_rank(cart_comm, coords, &rank);

    return rank;
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int dims[kNDim] = {0, 0};
    MPI_Dims_create(size, kNDim, dims);

    if (rank == 0)
    {
        std::cout << "dims created: x = " << dims[0] << " y = " << dims[1] << '\n';
    }

    int subdom_width = kGlobalWidth / dims[0];
    int subdom_height = kGlobalHeight / dims[1];

    MPI_Datatype row_t;
    MPI_Type_contiguous(subdom_width, MPI_INT, &row_t);
    MPI_Type_commit(&row_t);

    MPI_Datatype col_t;
    MPI_Type_vector(subdom_height, 1, subdom_width + 2, MPI_INT, &col_t);
    MPI_Type_commit(&col_t);
    
    MPI_Datatype block_t;
    {
        MPI_Datatype full_size_block_t;

        MPI_Type_vector(subdom_height, subdom_width, kGlobalWidth, MPI_INT, &full_size_block_t);
        MPI_Type_commit(&full_size_block_t);
        
        MPI_Type_create_resized(full_size_block_t, 0, 1, &block_t);
        MPI_Type_commit(&block_t);

        MPI_Type_free(&full_size_block_t);
    }

    MPI_Datatype displaced_block_t;
    MPI_Type_vector(subdom_height, subdom_width, subdom_width + 2, MPI_INT, &displaced_block_t);
    MPI_Type_commit(&displaced_block_t);

    Grid subdom(subdom_height + 2, subdom_width + 2);
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

    scatter_grid(global, subdom, dims, block_t, displaced_block_t);

    // print_grid(subdom);

    MPI_Comm cart_comm;
    const int periods[kNDim] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, kNDim, dims, periods, 1, &cart_comm);

    int own_coords[kNDim] = {-1, -1};
    MPI_Cart_coords(cart_comm, rank, kNDim, own_coords);

    int neighbors[8];

    neighbors[kTop] = GetCartCoords(cart_comm, own_coords, -1, 0);
    neighbors[kBottom] = GetCartCoords(cart_comm, own_coords, 1, 0);
    neighbors[kLeft] = GetCartCoords(cart_comm, own_coords, 0, -1);
    neighbors[kRight] = GetCartCoords(cart_comm, own_coords, 0, 1);
    
    neighbors[kTopLeft] = GetCartCoords(cart_comm, own_coords, -1, -1);
    neighbors[kTopRight] = GetCartCoords(cart_comm, own_coords, -1, 1);
    neighbors[kBottomLeft] = GetCartCoords(cart_comm, own_coords, 1, -1);
    neighbors[kBottomRight] = GetCartCoords(cart_comm, own_coords, 1, 1);

    sync_borders(subdom, neighbors, col_t, row_t, cart_comm);
    test_stop(subdom); // save gen 0 position

    // print_grid(subdom);

    int gen = 0;
    for (; gen < kMaxGenerations; ++gen)
    {
        Grid new_subdom = subdom;
        for (int i = 0; i < subdom.rows(); ++i)
        {
            for (int j = 1; j < subdom.cols() - 1; ++j)
            {
                int prev_i = i == 0 ? subdom.rows() - 1 : i - 1;
                int next_i = i == subdom.rows() - 1 ? 0 : i + 1;
                
                int neighbors_count =
                    subdom[prev_i][j - 1] + subdom[prev_i][j] + subdom[prev_i][j + 1] +
                    subdom[i]     [j - 1] + subdom[i][j + 1] +
                    subdom[next_i][j - 1] + subdom[next_i][j] + subdom[next_i][j + 1];
                
                    new_subdom[i][j] = (neighbors_count == 3) || 
                                       (subdom[i][j] && neighbors_count == 2);
                }
        }

        subdom = std::move(new_subdom);
        sync_borders(subdom, neighbors, col_t, row_t, cart_comm);

        if (test_stop(subdom))
        {
            break;
        }
    }

    gather_grid(global, subdom, dims, block_t, displaced_block_t);
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
    
    MPI_Type_free(&col_t);
    MPI_Type_free(&row_t);
    MPI_Type_free(&block_t);
    MPI_Type_free(&displaced_block_t);

    MPI_Finalize();
    
    return 0;
}