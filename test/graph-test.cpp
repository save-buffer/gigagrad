#include <catch2/catch_test_macros.hpp>
#include "src/graph.h"
#include "src/codegen.h"
#include "src/backend_scalar_c.h"

namespace gg = gigagrad;

TEST_CASE("TestXor", "[Graph]")
{
    gg::Graph graph;
    auto x = graph.AddInput(2);
    auto w1 = graph.AddInput({ 2, 2 });
    auto b1 = graph.AddWeight({ 2, 1 });
    auto w2 = graph.AddInput({ 1, 2 });
    auto b2 = graph.AddWeight(1);
    auto L1 = (w1 % x) + b1;
    auto L2 = (w2 % L1) + b2;
    auto result = L2.Compile<gg::codegen::BackendScalarC>();

    REQUIRE(L1.shape() == gg::Shape{2, 1});
    REQUIRE(result.shape == gg::Shape{1, 1});

    float x_data[] = { 1.0, 1.0 };
    float w1_data[] = { 1.0, -1.0, 1.0, -1.0 };
    float b1_data[] = { 0.5, -1.5 };
    float w2_data[] = { 1.0, 1.0 };
    float b2_data[] = { 1.5 };
    x.data() = x_data;
    w1.data() = w1_data;
    b1.data() = b1_data;
    w2.data() = w2_data;
    b2.data() = b2_data;
    result.Execute();
    REQUIRE(result.shape == gg::Shape{1, 1});
    REQUIRE(result.data[0] == 0.0f);

    x_data[0] = 0.0;
    result.Execute();
    REQUIRE(result.data[0] == 1.0f);
}

TEST_CASE("TestLogisticRegression", "[Graph]")
{
    gg::Graph graph;
    auto x = graph.AddInput({ 28, 28 }).reshape({ 28 * 28, 1 });
    auto w1 = graph.AddWeight({ 800, 28 * 28 });
    auto b1 = graph.AddWeight({ 800, 1 });
    auto z1 = (w1 % x) + b1;
    auto a2 = gg::sigmoid(z1);
    auto w2 = graph.AddWeight({ 10, 800 });
    auto b2 = graph.AddWeight({ 10, 1 });
    auto result = (w2 % a2) + b2;
    REQUIRE(result.shape() == gg::Shape{10, 1});
    result.Verify();
}

TEST_CASE("TestCreateGraph", "[Graph]")
{
    gigagrad::Graph graph;
    auto tensor1 = graph.AddInput({ 2, 2 });
    auto tensor2 = graph.AddWeight({ 2, 2 });
    auto addition = tensor1 + tensor2;

    REQUIRE(std::holds_alternative<gg::BinaryOp>(addition));
}
