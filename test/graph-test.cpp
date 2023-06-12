#include <catch2/catch_test_macros.hpp>
#include "src/graph.h"

TEST_CASE("TestLogisticRegression", "[Graph]")
{
    Gigagrad::Graph graph;
    auto x = graph.AddTensor({ 28, 28 });
    auto x_reshaped = x.reshape(28 * 28);
    auto w1 = graph.AddTensor(28 * 28);
    auto sig = 1.0f / (1.0f + exp(-w1 * x_reshaped));
    auto w2 = graph.AddTensor({ 10, 28 * 28 });
    auto result = (w2 % sig).max();
    result.Verify();
}

TEST_CASE("TestCreateGraph", "[Graph]")
{
    Gigagrad::Graph graph;
    auto tensor1 = graph.AddTensor({ 2, 2 });
    auto tensor2 = graph.AddTensor({ 2, 2 });
    auto addition = tensor1 + tensor2;

    REQUIRE(std::holds_alternative<Gigagrad::Tensor>(tensor1));
    REQUIRE(std::holds_alternative<Gigagrad::Tensor>(tensor2));
    REQUIRE(std::holds_alternative<Gigagrad::BinaryOp>(addition));
    REQUIRE(&std::get<Gigagrad::BinaryOp>(addition).x == &tensor1);
}
