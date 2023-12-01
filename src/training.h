#pragma once

#include "graph.h"

namespace gigagrad
{

struct Trainer
{
    struct NodeWithTrainingData
    {
        GraphNode *node;
        float *data;
    };

    void InitGraphIfNeeded(GraphNode &target)
    {
        Graph &target_graph = GetGraph(target);
        if(this->graph)
        {
            if(&target_graph != this->graph)
                throw std::domain_error("Trainer initialized with a different graph!");
            return;
        }
        this->graph = &target_graph;
        this->inputs.reserve(this->graph->inputs.size());
    }

    Trainer &Target(GraphNode &target, float *outputs)
    {
        InitGraphIfNeeded(target);
        this->target = { &target, outputs };
        return *this;
    }

    Trainer &TrainingData(GraphNode &input, float *inputs)
    {
        InitGraphIfNeeded(input);
        this->inputs.push_back({ &input, inputs });
        return *this;
    }

    Trainer &NumTrainingPoints(size_t num_points)
    {
        this->num_points = num_points;
        return *this;
    }

    void Train();
    void Validate();

    Graph *graph;
    std::optional<size_t> num_points = std::nullopt;
    std::vector<NodeWithTrainingData> inputs;
    NodeWithTrainingData target;
};

}
