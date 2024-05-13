#include "src/graph.h"
#include "src/backend_scalar_c.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <random>

namespace gg = gigagrad;

constexpr size_t MatrixSize = 512;
constexpr size_t NumIterations = 100;

void FillRandom(std::vector<float> &x)
{
    static std::default_random_engine e(0);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for(auto &v : x)
        v = dist(e);
}

int main()
{
    std::vector<float> A(MatrixSize * MatrixSize);
    std::vector<float> B(MatrixSize * MatrixSize);

    FillRandom(A);
    FillRandom(B);

    gg::Graph graph;
    auto a = graph.AddInput({ MatrixSize, MatrixSize });
    auto b = graph.AddInput({ MatrixSize, MatrixSize });
    auto matmul = a % b;
    auto result = matmul.Compile<gg::codegen::BackendScalarC>();

    a.data() = A.data();
    b.data() = B.data();

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < NumIterations; i++)
    {
        result.Execute();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        / static_cast<double>(NumIterations);

    printf("ScalarC: %.4fms\n", duration.count());

    return 0;
}
