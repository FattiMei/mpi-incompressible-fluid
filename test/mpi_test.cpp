#include <cassert>
#include <iostream>
#include <vector>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#pragma GCC diagnostic pop

int main(int argc, char *argv[]) {
  int rank;
  int size;

  // Initialize the MPI environment.
  MPI_Init(&argc, &argv);
  
  // Get the rank of the current processor.
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the total number of processors.
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Initialize some data to be sent and received.
  std::vector<int> data_send({rank, 10*rank, 20*rank, 
                              30*rank, 40*rank, 50*rank,
                              60*rank, 70*rank, 80*rank});
  std::vector<int> data_receive({-1, -1, -1, 
                                 -1, -1, -1,
                                 -1, -1, -1});

  MPI_Datatype mpi_type1;
  MPI_Type_contiguous(4, MPI_INT, &mpi_type1);
  MPI_Type_commit(&mpi_type1);

  MPI_Datatype mpi_type2;
  MPI_Type_vector(2, 2, 4, MPI_INT, &mpi_type2);
  MPI_Type_commit(&mpi_type2);

  MPI_Request request;
  MPI_Isend((int*) data_send.data() + 1, 1, mpi_type2, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request);
  //MPI_Isend((int*) data_send.data() + 5, 1, mpi_type1, (rank - 1 + size) % size, 1, MPI_COMM_WORLD, &request);

  MPI_Status status;
  MPI_Recv((int*) data_receive.data() + 3, 1, mpi_type1, (rank + 1 + size) % size, 0, MPI_COMM_WORLD, &status);
  //MPI_Recv((int*) data_receive.data() + 5, 1, mpi_type1, (rank + 1 + size) % size, 1, MPI_COMM_WORLD, &status);

  std::cout << "My rank: " << rank << ", my data: " << data_receive[0] << ", " << data_receive[1] << ", " << data_receive[2] << ", " 
                                                    << data_receive[3] << ", " << data_receive[4] << ", " << data_receive[5] << ", " 
                                                    << data_receive[6] << ", " << data_receive[7] << ", " << data_receive[8] << std::endl;

  // Finalize the MPI environment.
  MPI_Finalize();
}