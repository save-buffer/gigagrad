#include "training.h"

using namespace gigagrad;

#if 0
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
        return sum(Differentiate(g, y.x, dx), y.dims, y.keepdim);
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
    for(GraphNode &weight : this->graph->weights)
    {
        const Tensor &x = std::get<Tensor>(weight);
        gradients[i] = &Differentiate(backwards_graph, g, loss, x);
    }
}
#endif

void Differentiate(Graph &graph, const GraphNode &t, const GraphNode &seed);

void Differentiate(Graph &graph, const Tensor &t, const GraphNode &seed)
{
    // Add it to the list of tensors to compile
}

void Differentiate(Graph &graph, const Immediate &i, const GraphNode &seed)
{
    // ∇c = 0, where c is a constant
    return;
}

void Differentiate(Graph &graph, const UnaryOp &u, const GraphNode &seed)
{
    switch(u.type)
    {
    case UnaryOpType::EXP:
        // ∇(exp(x)) = { s * exp(x) * ∂x }
        Differentiate(graph, u.x, seed * u);
        break;
    case UnaryOpType::LOG:
        // ∇(log(x)) = { s/x * ∂x }
        Differentiate(graph, u.x, seed / u.x);
        break;
    case UnaryOpType::SIN:
        // ∇(sin(x)) = { s * cos(x)∂x }
        Differentiate(graph, u.x, cos(u.x) * seed);
        break;
    default:
        throw std::runtime_error("Unimplemented operation");
    }
}

void Differentiate(Graph &graph, const BinaryOp &b, const GraphNode &seed)
{
    switch(b.type)
    {
    case BinaryOpType::ADD:
        // ∇(x + y) = { ∂x, ∂y }
        Differentiate(graph, b.x, seed);
        Differentiate(graph, b.y, seed);
        break;
    case BinaryOpType::SUB:
        // ∇(x - y) = { ∂x, -∂y }
        Differentiate(graph, b.x, seed);
        Differentiate(graph, -b.y, seed);
        break;
    case BinaryOpType::MUL:
        // ∇(x * y) = { ys * ∂x, xs * ∂y }
        Differentiate(graph, b.x, b.y * seed);
        Differentiate(graph, b.y, b.x * seed);
        break;
    case BinaryOpType::DIV:
        // ∇(x / y) = { s/y * ∂x, -s*x/(∂y)^2 }
        Differentiate(graph, b.x, seed / b.y);
        Differentiate(graph, b.y * b.y, -seed * b.x);
        break;
    case BinaryOpType::POW:
        // ∇(x^y) = { syx^(y - 1) * ∂x, s * log(x) * x^y * ∂y }
        Differentiate(graph, b.x, seed * b.y * pow(b.x, b.y - 1));
        Differentiate(graph, b.y, log(b.x) * pow(b.x, b.y));
        break;
    case BinaryOpType::CMP:
        // ∇(x == y) = { s * (x == y ? 1 : 0), s * (a == b ? 1 : 0) }
        Differentiate(graph, b.x, seed * (b.x == b.y));
        Differentiate(graph, b.y, seed * (b.x == b.y));
        break;
    case BinaryOpType::MAX:
        // ∇(max(x, y)) = { s * (x > y), s * (y >= x) }
        Differentiate(graph, b.x, seed * (b.x > b.y));
        Differentiate(graph, b.y, seed * (b.y >= b.x));
        break;
    }
}

void Differentiate(Graph &graph, const ReduceOp &r, const GraphNode &seed)
{
    switch(r.type)
    {
    case ReduceOpType::SUM:
        Differentiate(graph, r.x, seed);
        break;
    case ReduceOpType::MAX:
        break;
    }
}

void Differentiate(Graph &graph, const GraphNode &node, const GraphNode &seed)
{
    std::visit([&](auto &&x) { Differentiate(graph, x, seed); }, node);
}

void Train(Graph &graph, const GraphNode &output)
{
    const GraphNode &model_output = graph.AddInput(output.shape());
    const GraphNode &training_example = graph.AddInput(output.shape()); 
    const GraphNode &error = model_output - training_example;
    const GraphNode &loss = sum(error % error);
    const GraphNode &seed = graph.Immediate(1.0f);
    Differentiate(graph, loss, seed);
}
