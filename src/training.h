#pragma once

#include "graph.h"
#include "codegen.h"

namespace gigagrad
{

struct TrainingContext
{
    float *loss;
    std::unique_ptr<codegen::Backend> backend;

    void Execute() { backend->Execute(); }
};

// TODO: Allow dynamic learning rate
TrainingContext CompileTrainingGraph(
    nn::Module &network,
    GraphNodeHandle model_output,
    std::unique_ptr<codegen::Backend> backend,
    float learning_rate = 0.1f);

template <typename TBackend>
TrainingContext CompileTrainingGraph(nn::Module &network, GraphNodeHandle loss, float learning_rate = 0.1f)
{
    return CompileTrainingGraph(network, loss, std::make_unique<TBackend>(), learning_rate);
}

GraphNodeHandle L2Loss(GraphNodeHandle output, GraphNodeHandle training_example)
{
    GraphNodeHandle error = output - training_example;
    GraphNodeHandle loss = sum(error * error);
    return loss;
}

GraphNodeHandle CrossEntropyLoss(GraphNodeHandle output, GraphNodeHandle training_example)
{
    dim_t batch_size = output.shape().size() <= 2 ? 1 : output.shape()[0];
    GraphNodeHandle lg_sm = log_softmax(output, dim_t{-2});
    GraphNodeHandle cross_entropy = lg_sm * training_example;
    GraphNodeHandle loss = (1.0f / batch_size) * sum(cross_entropy);
    return loss;
}

}
