#include<iostream>
#include<random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include<chrono>
#include<numeric>
#include <Eigen/Dense>
#include <mpi.h>

using namespace std;


int** convertMatrix(vector<int> Matrix, int size)
{
    int** newMatrix = new int* [size];
    for (size_t i = 0; i < size; i++)
    {
        newMatrix[i] = new int[size];
        for (size_t j = 0; j < size; j++)
        {
            newMatrix[i][j] = Matrix[j + i * size];
        }
    }
    return newMatrix;
}

vector<int> getVector(int size, bool isEmthy)
{
    vector<int> matrix(size *size);
    if (!isEmthy)
    {
        for (int i = 0; i < size * size; i++)
        {
            matrix[i] = rand() % 200;
        }
    }
    return matrix;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //std::ofstream outFile("./performance_MPI.csv", std::ios::out | std::ios::app);
    ofstream outFile;

    if (rank == 0) {
        outFile.open("./results.json", ios::out);
        outFile << "{ \"results\": [\n";
    }

    int threads = world_size;

    for (int matrixSize = 100; matrixSize <= 1000; matrixSize += 100)
    {   
        MPI_Barrier(MPI_COMM_WORLD);

        vector<int> matrix1 = getVector(matrixSize, true);
        vector<int> matrix2 = getVector(matrixSize, true);
        vector<int> resultMatrix = getVector(matrixSize, true);
        if (rank == 0)
        {
            cout << "elements count: " << matrixSize << endl;
            matrix1 = getVector(matrixSize, false);
            matrix2 = getVector(matrixSize, false);

            std::vector<double> averageDurations;
            long long totalDuration = 0;

            MPI_Bcast(matrix1.data(), matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(matrix2.data(), matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast(matrix1.data(), matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(matrix2.data(), matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
        }
        

        MPI_Barrier(MPI_COMM_WORLD);
        const int times = 10;


        std::vector<double> durations;
        for (int k = 0; k < times; ++k)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = chrono::high_resolution_clock::now();
            vector<int> subMatrix1;
            vector<int> resultSubMatrix;


            int rowsPerProcess = matrixSize / world_size;
            subMatrix1.resize(rowsPerProcess * matrixSize);
            resultSubMatrix.resize(rowsPerProcess * matrixSize);

            MPI_Scatter(matrix1.data(), rowsPerProcess * matrixSize, MPI_INT,
                subMatrix1.data(), rowsPerProcess * matrixSize, MPI_INT,
                0, MPI_COMM_WORLD);

            for (int i = 0; i < rowsPerProcess; ++i)
            {
                for (int j = 0; j < matrixSize; ++j)
                {
                    resultSubMatrix[i * matrixSize + j] = 0;
                    for (int k = 0; k < matrixSize; k++)
                    {
                        resultSubMatrix[i * matrixSize + j] += subMatrix1[i * matrixSize + k] * matrix2[k * matrixSize + j];
                    }
                }
            }

            MPI_Gather(resultSubMatrix.data(), rowsPerProcess * matrixSize, MPI_INT,
                resultMatrix.data(), rowsPerProcess * matrixSize, MPI_INT,
                0, MPI_COMM_WORLD);

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            durations.push_back(duration);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0)
            {
                //outFile << matrixSize << "," << world_size << "," << totalTime << "\n";
                int** resMatrix = convertMatrix(resultMatrix, matrixSize);
                int** m1 = convertMatrix(matrix1, matrixSize);
                int** m2 = convertMatrix(matrix2, matrixSize);


                for (int d = 0; d < matrixSize; d++)
                {
                    delete[] m1[d];
                    delete[] m2[d];
                    delete[] resMatrix[d];

                }
                delete[] m1;
                delete[] m2;
                delete[] resMatrix;
            }

        }
        if (rank == 0)
        {
            double mean = accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
            outFile << "  { \"MatrixSize\": " << matrixSize << ", \"Time(ms)\": " << mean << " }";
            if (matrixSize != 1000) outFile << ",\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
    outFile << "\n]}\n";
    outFile.close();

    MPI_Finalize();

    return 0;
}