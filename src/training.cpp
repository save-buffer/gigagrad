#include "training.h"

using namespace gigagrad;

struct BackpropCtx
{
    struct WeightGradients
    {
        GraphNodeHandle weight;
        GraphNodeHandle gradient;
    };

    std::vector<WeightGradients> gradients;
    codegen::Program program;
};

void Differentiate(BackpropCtx &ctx, const GraphNode &t, const GraphNode &seed);

void Differentiate(BackpropCtx &ctx, GraphNodeHandle node, const Tensor &t, GraphNodeHandle seed)
{
    ctx.gradients.emplace_back({ node, seed });
}

void Differentiate(BackpropCtx &ctx, GraphNodeHandle, const Immediate &i, const GraphNode &seed)
{
    // ∇c = 0, where c is a constant
    return;
}

void Differentiate(BackpropCtx &ctx, GraphNodeHandle, const UnaryOp &u, const GraphNode &seed)
{
    switch(u.type)
    {
    case UnaryOpType::EXP:
        // ∇(exp(x)) = { s * exp(x) * ∂x }
        Differentiate(ctx, u.x, seed * u);
        break;
    case UnaryOpType::LOG:
        // ∇(log(x)) = { s/x * ∂x }
        Differentiate(ctx, u.x, seed / u.x);
        break;
    case UnaryOpType::SIN:
        // ∇(sin(x)) = { s * cos(x)∂x }
        Differentiate(ctx, u.x, cos(u.x) * seed);
        break;
    default:
        throw std::runtime_error("Unimplemented operation");
    }
}

void Differentiate(BackpropCtx &ctx, GraphNodeHandle, const BinaryOp &b, const GraphNode &seed)
{
    switch(b.type)
    {
    case BinaryOpType::ADD:
        // ∇(x + y) = { ∂x, ∂y }
        Differentiate(ctx, b.x, seed);
        Differentiate(ctx, b.y, seed);
        break;
    case BinaryOpType::SUB:
        // ∇(x - y) = { ∂x, -∂y }
        Differentiate(ctx, b.x, seed);
        Differentiate(ctx, -b.y, seed);
        break;
    case BinaryOpType::MUL:
        // ∇(x * y) = { ys * ∂x, xs * ∂y }
        Differentiate(ctx, b.x, b.y * seed);
        Differentiate(ctx, b.y, b.x * seed);
        break;
    case BinaryOpType::DIV:
        // ∇(x / y) = { s/y * ∂x, -s*x/(∂y)^2 }
        Differentiate(ctx, b.x, seed / b.y);
        Differentiate(ctx, b.y * b.y, -seed * b.x);
        break;
    case BinaryOpType::POW:
        // ∇(x^y) = { syx^(y - 1) * ∂x, s * log(x) * x^y * ∂y }
        Differentiate(ctx, b.x, seed * b.y * pow(b.x, b.y - 1));
        Differentiate(ctx, b.y, log(b.x) * pow(b.x, b.y));
        break;
    case BinaryOpType::CMP:
        // ∇(x == y) = { s * (x == y ? 1 : 0), s * (a == b ? 1 : 0) }
        Differentiate(ctx, b.x, seed * (b.x == b.y));
        Differentiate(ctx, b.y, seed * (b.x == b.y));
        break;
    case BinaryOpType::MAX:
        // ∇(max(x, y)) = { s * (x > y), s * (y >= x) }
        Differentiate(ctx, b.x, seed * (b.x > b.y));
        Differentiate(ctx, b.y, seed * (b.y >= b.x));
        break;
    }
}

void Differentiate(BackpropCtx &ctx, GraphNodeHandle, const ReduceOp &r, const GraphNode &seed)
{
    switch(r.type)
    {
    case ReduceOpType::SUM:
        Differentiate(ctx, r.x, seed);
        break;
    case ReduceOpType::MAX:
        // TODO: Support this once we have argmax
        throw std::runtime_error("Gradients involving Max reduction not implemented");
    }
}

void Differentiate(BackpropCtx &ctx, GraphNodeHandle node, GraphNodeHandle seed)
{
    node.Visit([&](auto &&x) { Differentiate(ctx, node, x, seed); });
}

void Train(Graph &graph, const GraphNodeHandle output)
{
    GraphNodeHandle model_output = graph.AddInput(output.shape());
    GraphNodeHandle training_example = graph.AddInput(output.shape()); 
    GraphNodeHandle error = model_output - training_example;
    GraphNodeHandle loss = sum(error % error);
    GraphNodeHandle seed = graph.Immediate(1.0f);
    BackpropCtx ctx;
    Differentiate(ctx, loss, seed);
    CodegenNode(ctx.program, loss);
    for(size_t i = 0; i < ctx.gradients.size(); i++)
    {
        CodegenNode(ctx.program, ctx.gradients[i].gradient);
    }
    
}
