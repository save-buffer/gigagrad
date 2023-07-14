#include <catch2/catch_test_macros.hpp>
#include "src/graph.h"
#include "src/codegen.h"

TEST_CASE("TestLogisticRegression", "[Graph]")
{
    Gigagrad::Graph graph;
    auto x = graph.AddInput({ 28, 28 }).reshape({ 28 * 28, 1 });
    auto w1 = graph.AddWeight({ 800, 28 * 28 });
    auto b1 = graph.AddWeight({ 800, 1 });
    auto z1 = (w1 % x) + b1;
    auto a2 = sigmoid(z1);
    auto w2 = graph.AddWeight({ 10, 800 });
    auto b2 = graph.AddWeight({ 10, 1 });
    auto result = (w2 % a2) + b2;
    REQUIRE(result.shape() == Gigagrad::Shape{10, 1});
    result.Verify();
    Gigagrad::PrintCodegenNode(result);
}

TEST_CASE("TestCreateGraph", "[Graph]")
{
    Gigagrad::Graph graph;
    auto tensor1 = graph.AddInput({ 2, 2 });
    auto tensor2 = graph.AddWeight({ 2, 2 });
    auto addition = tensor1 + tensor2;

    REQUIRE(std::holds_alternative<Gigagrad::Tensor>(tensor1));
    REQUIRE(std::holds_alternative<Gigagrad::Tensor>(tensor2));
    REQUIRE(std::holds_alternative<Gigagrad::BinaryOp>(addition));
    REQUIRE(&std::get<Gigagrad::BinaryOp>(addition).x == &tensor1);
}
