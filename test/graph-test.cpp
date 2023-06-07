#include <catch2/catch_test_macros.hpp>
#include "src/graph.h"

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
