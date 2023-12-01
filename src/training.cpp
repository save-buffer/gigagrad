#include "training.h"

using namespace gigagrad;

const GraphNode &Differentiate(Graph &g, const GraphNode &y, const Tensor &dx);

const GraphNode &Differentiate(Graph &g, const Tensor &y, const Tensor &dx)
{
    if(&y == &dx)
        return g.AddNode(Immediate{g, 1.0f});
    return g.AddNode(Immediate{g, 0.0f});
}

const GraphNode &Differentiate(Graph &g, const Immediate &y, const Tensor &dx)
{
    return g.AddNode(Immediate{g, 0.0f});
}

const GraphNode &Differentiate(Graph &g, const UnaryOp &y, const Tensor &dx)
{
    switch(y.type)
    {
    case UnaryOpType::NOP:
    {
        return Differentiate(g, y.x, dx);
    }
    case UnaryOpType::EXP:
    {
        return Differentiate(g, y.x, dx) * exp(y.x);
    }
    case UnaryOpType::LOG:
    {
        return Differentiate(g, y.x, dx) / y.x;
    }
    case UnaryOpType::CAST:
    {
    }
    case UnaryOpType::SIN:
    {
        return sin(y.x + (3.14159265f/2.0f)) * Differentiate(g, y.x, dx);
    }
    }
}

const GraphNode &Differentiate(Graph &g, const BinaryOp &y, const Tensor &dx)
{
    switch(y.type)
    {
    case BinaryOpType::ADD:
    {
        return Differentiate(g, y.x, dx) + Differentiate(g, y.y, dx);
    }
    case BinaryOpType::SUB:
    {
        return Differentiate(g, y.x, dx) - Differentiate(g, y.y, dx);
    }
    case BinaryOpType::MUL:
    {
        const GraphNode &dudx = Differentiate(g, y.x, dx);
        const GraphNode &dvdx = Differentiate(g, y.y, dx);
        return y.x * dvdx + y.y * dudx;
    }
    case BinaryOpType::DIV:
    {
        const GraphNode &dudx = Differentiate(g, y.x, dx);
        const GraphNode &dvdx = Differentiate(g, y.y, dx);
        return (y.x * dvdx - y.y * dudx) / (y.x * y.y);
    }
    case BinaryOpType::POW:
    {
        return y.y * pow(y.x, y.y - 1) * Differentiate(g, y.x, dx);
    }
    case BinaryOpType::CMP:
    {
    }
    case BinaryOpType::MAX:
    {
    }
    }
}

const GraphNode &Differentiate(Graph &g, const ReduceOp &y, const Tensor &dx)
{
    switch(y.type)
    {
    case ReduceOpType::SUM:
    {
        return sum(Differentiate(g, y.x, dx), y.shape, y.keepdim);
    }
    case ReduceOpType::MAX:
    {
        
    }
    }
}

const GraphNode &Differentiate(Graph &g, const ReshapeOp &y, const Tensor &dx)
{
    const GraphNode &dudx = Differentiate(g, y.x, dx);
    return dudx.reshape(y.shape);
}

const GraphNode &Differentiate(Graph &g, const PermuteOp &y, const Tensor &dx)
{
    const GraphNode &dudx = Differentiate(g, y.x, dx);
    return dudx.permute(y.dims);
}

const GraphNode &Differentiate(Graph &g, const GraphNode &y, const Tensor &dx)
{
    return std::visit([&](auto &&x) -> const GraphNode &{ return Differentiate(g, y, dx); }, y);
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
    auto example = this->graph->AddInput(target.node->shape());
    auto error = *(target.node) - example;
    auto loss = sum(error % error);

    Graph backwards_graph;
    std::vector<const GraphNode *> gradients;
    for(GraphNode &weight : this->graph.weights)
    {
        const Tensor &x = std::get<Tensor>(weight);
        gradients[i] = &Differentiate(backwards_graph, g, loss, x);
    }
    
}
