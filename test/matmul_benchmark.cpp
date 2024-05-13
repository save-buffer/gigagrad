#include "src/graph.h"
#include "src/backend_scalar_c.h"
#include "src/backend_metal.h"

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

    a.data() = A.data();
    b.data() = B.data();

    auto scalar = matmul.Compile<gg::codegen::BackendScalarC>();
    auto metal = matmul.Compile<gg::codegen::BackendMetal>();

    auto start_scalar = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < NumIterations; i++)
    {
        scalar.Execute();
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();

    auto start_metal = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < NumIterations; i++)
    {
        metal.Execute();
    }
    auto end_metal = std::chrono::high_resolution_clock::now();

    auto duration_scalar = std::chrono::duration_cast<std::chrono::milliseconds>(end_scalar - start_scalar)
        / static_cast<double>(NumIterations);

    auto duration_metal = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal)
        / static_cast<double>(NumIterations);

    printf("Scalar: %.4fms\n", duration_scalar.count());
    printf("Metal: %.4fms\n", duration_metal.count());

    return 0;
}
