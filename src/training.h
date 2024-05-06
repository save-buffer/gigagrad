#pragma once

#include "graph.h"
#include "codegen.h"

namespace gigagrad
{

struct TrainingContext
{
    float *loss;
    float *&training_example;
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
TrainingContext CompileTrainingGraph(nn::Module &network, GraphNodeHandle model_output, float learning_rate = 0.1f)
{
    return CompileTrainingGraph(network, model_output, std::make_unique<TBackend>(), learning_rate);
}

}
