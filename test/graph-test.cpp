#include <catch2/catch_test_macros.hpp>
#include "src/graph.h"
#include "src/codegen.h"
#include "src/backend_scalar_c.h"
#include "src/training.h"

namespace gg = gigagrad;

TEST_CASE("TestTrain", "[Train]")
{
    gg::nn::Module network;
    auto x = network.AddInput(4);
    auto w = network.AddWeight(4);
    auto L1 = w - x;
    gg::TrainingContext ctx = gg::CompileTrainingGraph<gg::codegen::BackendScalarC>(network, L1);
    float x_data[] = { 1.0, 2.0, 3.0, 4.0 };
    float w_data[] = { -0.1, 0.1, -0.001, 0.0001 };
    float training_example_data[] = { 0.0, 0.0, 0.0, 0.0 };
    x.data() = x_data;
    w.data() = w_data;
    ctx.training_example = training_example_data;
    for(int i = 0; i < 10; i++)
    {
        ctx.Execute();
        printf("%.6f\n", ctx.loss[0]);
    }
}

TEST_CASE("TestXor", "[Graph]")
{
    gg::Graph graph;
    auto x = graph.AddInput(2);
    auto w1 = graph.AddInput({ 2, 2 });
    auto w2 = graph.AddInput({ 1, 2 });
    auto b1 = graph.AddInput({ 2, 1 });
    auto L1 = (w1 % x) > b1;
    auto L2 = (w2 % L1) > 1.5f;
    auto result = L2.Compile<gg::codegen::BackendScalarC>();

    REQUIRE(L1.shape() == gg::Shape{2, 1});
    REQUIRE(L2.shape() == gg::Shape{1, 1});

    float x_data[] = { 1.0, 1.0 };
    float w1_data[] = { 1.0, 1.0, -1.0, -1.0 };
    float b1_data[] = { 0.5, -1.5 };
    float w2_data[] = { 1.0, 1.0 };
    x.data() = x_data;
    w1.data() = w1_data;
    b1.data() = b1_data;
    w2.data() = w2_data;

    for(bool x1 : { false, true })
    {
        x_data[0] = x1 ? 1.0f : 0.0f;
        for(bool x2 : { false, true })
        {
            x_data[1] = x2 ? 1.0f : 0.0f;
            result.Execute();

            float expected = (x1 ^ x2) ? 1.0f : 0.0f;
            REQUIRE(result.data[0] == expected);
        }
    }
}

TEST_CASE("TestLogisticRegression", "[Graph]")
{
    gg::Graph graph;
    auto x = graph.AddInput({ 28, 28 }).reshape({ 28 * 28, 1 });
    auto w1 = graph.AddInput({ 800, 28 * 28 });
    auto b1 = graph.AddInput({ 800, 1 });
    auto z1 = (w1 % x) + b1;
    auto a2 = gg::sigmoid(z1);
    auto w2 = graph.AddInput({ 10, 800 });
    auto b2 = graph.AddInput({ 10, 1 });
    auto result = (w2 % a2) + b2;
    REQUIRE(result.shape() == gg::Shape{10, 1});
}

TEST_CASE("TestCreateGraph", "[Graph]")
{
    gigagrad::Graph graph;
    auto tensor1 = graph.AddInput({ 2, 2 });
    auto tensor2 = graph.AddInput({ 2, 2 });
    auto addition = tensor1 + tensor2;

    REQUIRE(addition->Kind() == gg::GraphNode::Kind::BinaryOp);
}
