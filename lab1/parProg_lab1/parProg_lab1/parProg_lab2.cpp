#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <omp.h>

using namespace std;

bool ckeckMultiplying(int** matrix, int** secondMatrix, const int& sizeTotal, int** handResultMatrix)
{
    Eigen::MatrixXi eigenMatrixA(sizeTotal, sizeTotal), eigenMatrixB(sizeTotal, sizeTotal);
    for (int i = 0; i < sizeTotal; ++i) {
        for (int j = 0; j < sizeTotal; ++j) {
            eigenMatrixA(i, j) = matrix[i][j];
            eigenMatrixB(i, j) = secondMatrix[i][j];
        }
    }
    Eigen::MatrixXi resultMatrix = eigenMatrixA * eigenMatrixB;

    for (int i = 0; i < sizeTotal; ++i) {
        for (int j = 0; j < sizeTotal; ++j) {
            if (handResultMatrix[i][j] != resultMatrix(i, j))
            {
                return false;
            }
        }
    }
    return true;

}

int** CreateMatrix(int sizem)
{
    int** matrix = new int* [sizem];

    for (int i = 0; i < sizem; i++)
    {
        matrix[i] = new int[sizem];
    }

    std::srand(std::time(0));

    for (int i = 0; i < sizem; ++i)
    {
        for (int j = 0; j < sizem; ++j)
        {
            matrix[i][j] = std::rand() % 200;
        }
    }
    //for (int i = 0; i < sizem; ++i)
    //{
    //    for (int j = 0; j < sizem; ++j)
    //    {
    //        std::cout << matrix[i][j];
    //    }
    //}
    return matrix;

}

void WriteMatrixToFile(int** matrix, const int& sizeTotal, const std::string fileName)
{
    std::ofstream file(fileName);
    for (int i = 0; i < sizeTotal; ++i) {
        for (int j = 0; j < sizeTotal; ++j) {
            file << matrix[i][j];
            if (j < sizeTotal - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

int** ReadMatrixFromFile(const int& sizeTotal, const std::string fileName)
{
    std::ifstream file(fileName);
    std::string line;
    int** matrix = new int* [sizeTotal];

    for (int i = 0; i < sizeTotal && getline(file, line); ++i) {
        matrix[i] = new int[sizeTotal];
        std::istringstream lineStream(line);
        std::string cell;
        int j = 0;

        while (getline(lineStream, cell, ',') && j < sizeTotal) {
            matrix[i][j] = std::stoi(cell);
            j++;
        }
    }

    file.close();
    return matrix;
}

int** MultiplyMatrices(int** matrix, int** secondMatrix, const int& sizeTotal, double& totalTime, bool& isCorrect)
{
    int count = 10;
    totalTime = 0;
    for (int cunt = 0; cunt < count; cunt++)
    {
        clock_t start = clock();
        
        int** resultMatrix = new int* [sizeTotal];

        #pragma omp parallel for
        for (int i = 0; i < sizeTotal; ++i) {
            resultMatrix[i] = new int[sizeTotal];
            for (int j = 0; j < sizeTotal; ++j) {
                resultMatrix[i][j] = 0;
                for (int k = 0; k < sizeTotal; ++k) {
                    resultMatrix[i][j] += matrix[i][k] * secondMatrix[k][j];
                }
            }
        }
        clock_t end = clock();
        totalTime += double(end - start) / CLOCKS_PER_SEC;
        if (isCorrect)
        {
            isCorrect = ckeckMultiplying(matrix, secondMatrix, sizeTotal, resultMatrix);
            cout << isCorrect << "\n";
        }

        for (int i = 0; i < sizeTotal; i++)
        {
            delete[] resultMatrix[i];
        }
        delete[] resultMatrix;
    }

    totalTime = totalTime / count;

    int** resultMatrix = new int* [sizeTotal];
    for (int i = 0; i < sizeTotal; ++i) {
        resultMatrix[i] = new int[sizeTotal];
        for (int j = 0; j < sizeTotal; ++j) {
            resultMatrix[i][j] = 0;
            for (int k = 0; k < sizeTotal; ++k) {
                resultMatrix[i][j] += matrix[i][k] * secondMatrix[k][j];
            }
        }
    }

    if (isCorrect)
    {
        isCorrect = ckeckMultiplying(matrix, secondMatrix, sizeTotal, resultMatrix);
        cout << isCorrect << "\n";
    }

    return resultMatrix;
}

void writeJsonToFile(const std::string& filePath, bool isCorrect, double timeTotal, const int& sizeTotal)
{
    int elementsCount = sizeTotal * sizeTotal;
    std::string json = "{\n";
    json += "  \"ItemsCount\": " + std::to_string(elementsCount) + ",\n";
    json += "  \"IsMultiplicationCorrect\": " + std::string(isCorrect ? "true" : "false") + ",\n";
    json += "  \"DurationSeconds\": " + std::to_string(timeTotal) + "\n";
    json += "}\n";

    std::ofstream outFile(filePath);
    if (outFile.is_open()) {
        outFile << json;
        outFile.close();
    }
    else {
        std::cerr << "Unable to open file for writing JSON data.\n";
    }
}

int main()
{
    /*cout
        << "Processors: " << omp_get_num_procs()
        << ", Max threads: " << omp_get_max_threads()
        << ", threads: " << omp_get_num_threads()
        << std::endl;*/
    int threadsNumber = 32;
    omp_set_num_threads(threadsNumber);
    bool isCorrect = true;

    //const char* firstMatrixFileName = "firstMatrix.csv";
    //const char* secondMatrixFileName = "secondMatrix.csv";
    const std::string resultFileName = "result.json";
    const std::string resultMatrixFileName = "resultMatrix.csv";
    const std::string firstMatrixFileName = "firstMatrix.csv";
    const std::string secondMatrixFileName = "secondMatrix.csv";
    int sizeTotal = 800;
    double totalTime = 0;

    int** matrix = CreateMatrix(sizeTotal);
    int** secondMatrix = CreateMatrix(sizeTotal);
    WriteMatrixToFile(matrix, sizeTotal, firstMatrixFileName);
    WriteMatrixToFile(secondMatrix, sizeTotal, secondMatrixFileName);
    //int** readedMatrix = ReadMatrixFromFile(sizeTotal, firstMatrixFileName);
    //int** secondReadedMatrix = ReadMatrixFromFile(sizeTotal, secondMatrixFileName);

    //omp_set_num_threads(threadsNumber);------------------------------------------------------------------------
    int** resultMatrix = MultiplyMatrices(matrix, secondMatrix, sizeTotal, totalTime, isCorrect);
    //bool isCorrect = ckeckMultiplying(matrix, secondMatrix, sizeTotal, resultMatrix);

    bool ckeckResult = ckeckMultiplying(matrix, secondMatrix, sizeTotal, resultMatrix);
    writeJsonToFile(resultFileName, isCorrect, totalTime, sizeTotal);
    WriteMatrixToFile(resultMatrix, sizeTotal, resultMatrixFileName);
    return 0;
}