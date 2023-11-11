#include "training.h"

using namespace gigagrad;

void Differentiate(const GraphNode &g);

void Differentiate(const Reduceop &r)
{
    
}

void Differentiate(const ReshapeOp &r)
{
    Differentiate(r.x);
}

void Differentiate(const PermuteOp &p)
{
    Differentiate(p.x);
}

void Differentiate(const GraphNode &g)
{
    return std::visit([&](auto &&x) { Differentiate(x); });
}

void Trainer::Validate()
{
    if(!this->graph)
        throw std::runtime_error("Tried to train without a target graph");
    if(!this->num_points)
        throw std::runtime_error("Number of training examples not specified");
    if(this->inputs.size() < this->graph->inputs.size())
        throw std::runtime_error("Training inputs not specified for all graph inputs");
    if(this->inputs.size() > this->graph->inputs.size())
        throw std::runtime_error("Too many training inputs specified for graph");
}

void Trainer::Train()
{
    this->Validate();
    auto example = this->graph->AddInput(*target->node.shape());
    auto error = *target->node - example;
    auto loss = sum(error % error);

    std::vector<Graph> derivatives;
}
