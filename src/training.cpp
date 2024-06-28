#include "training.h"
#include "codegen.h"

using namespace gigagrad;

GraphNodeHandle gigagrad::L2Loss(GraphNodeHandle output, GraphNodeHandle training_example)
{
    GraphNodeHandle error = output - training_example;
    GraphNodeHandle loss = sum(error * error);
    return loss;
}

GraphNodeHandle gigagrad::CrossEntropyLoss(GraphNodeHandle output, GraphNodeHandle training_example)
{
    dim_t batch_size = output.shape().size() <= 2 ? 1 : output.shape()[0];
    GraphNodeHandle lg_sm = log_softmax(output, dim_t{-2});
    GraphNodeHandle cross_entropy = lg_sm * training_example;
    GraphNodeHandle loss = (-1.0f / batch_size) * sum(cross_entropy);
    return loss;
}

struct Gradient
{
    GraphNodeHandle input;
    GraphNodeHandle gradient;
};

struct BackpropContext
{
    // Note: `gradients` contains gradients for all Tensors, not just weights.
    // Additional filtering is needed based on `first`.
    std::vector<Gradient> gradients;
    codegen::Program program;
};

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, GraphNodeHandle seed);

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, const Tensor &t, GraphNodeHandle seed)
{
    // If we're performing batched training
    if(seed.shape().size() > node.shape().size())
    {
        size_t num_example_dims = seed.shape().size() - node.shape().size();
        auto num_examples = std::accumulate(
            seed.shape().begin(),
            seed.shape().begin() + num_example_dims,
            1,
            std::multiplies{});
        Dims dims(num_example_dims);
        std::iota(dims.begin(), dims.end(), 0);
        seed = seed.sum(std::move(dims)) / static_cast<float>(num_examples);
    }
    ctx.gradients.push_back({ node, seed });
}

void Differentiate(BackpropContext &ctx, GraphNodeHandle, const Immediate &i, const GraphNodeHandle seed)
{
    // ∇c = 0, where c is a constant
    return;
}

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, const UnaryOp &u, GraphNodeHandle seed)
{
    switch(u.type)
    {
    case UnaryOpType::EXP:
        // ∇(exp(x)) = { s * exp(x) * ∂x }
        Differentiate(ctx, u.x, seed * node);
        break;
    case UnaryOpType::LOG:
        // ∇(log(x)) = { s/x * ∂x }
        Differentiate(ctx, u.x, seed / u.x);
        break;
    case UnaryOpType::SIN:
        // ∇(sin(x)) = { s * cos(x) * ∂x }
        Differentiate(ctx, u.x, seed * cos(u.x));
        break;
    case UnaryOpType::SQRT:
        // ∇(sqrt(x)) = { s * 1/(2 * sqrt(x)) * ∂x }
        Differentiate(ctx, u.x, seed / (2.0f * node));
        break;
    default:
        throw std::runtime_error("Unimplemented operation");
    }
}

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, const BinaryOp &b, GraphNodeHandle seed)
{
    switch(b.type)
    {
    case BinaryOpType::ADD:
        // ∇(x + y) = { s * ∂x, s * ∂y }
        Differentiate(ctx, b.x, seed);
        Differentiate(ctx, b.y, seed);
        break;
    case BinaryOpType::SUB:
        // ∇(x - y) = { s * ∂x, s * -1 * ∂y }
        Differentiate(ctx, b.x, seed);
        Differentiate(ctx, b.y, -seed);
        break;
    case BinaryOpType::MUL:
        // ∇(x * y) = { s * y * ∂x, s * x * ∂y }
        Differentiate(ctx, b.x, seed * b.y);
        Differentiate(ctx, b.y, seed * b.x);
        break;
    case BinaryOpType::DIV:
        // ∇(x / y) = { s/y * ∂x, -s*x/y^2 * ∂y }
        Differentiate(ctx, b.x, seed / b.y);
        Differentiate(ctx, b.y, -seed * b.x / (b.y * b.y));
        break;
    case BinaryOpType::POW:
        // ∇(x^y) = { syx^(y - 1) * ∂x, s * log(x) * x^y * ∂y }
        Differentiate(ctx, b.x, seed * b.y * pow(b.x, b.y - 1));
        Differentiate(ctx, b.y, seed * log(b.x) * node);
        break;
    case BinaryOpType::CMP:
        // ∇(x == y) = { s * (x == y ? 1 : 0), s * (a == b ? 1 : 0) }
        Differentiate(ctx, b.x, seed * node);
        Differentiate(ctx, b.y, seed * node);
        break;
    case BinaryOpType::MAX:
        // ∇(max(x, y)) = { s * (x > y), s * (y >= x) }
        Differentiate(ctx, b.x, seed * (b.x > b.y));
        Differentiate(ctx, b.y, seed * (b.y >= b.x));
        break;
    }
}

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, const ReduceOp &r, GraphNodeHandle seed)
{
    // Reinsert 1's if we don't have keepdim so that broadcasting semantics work
    if(!r.keepdim && !seed.shape().empty())
    {
        Shape shape(r.x.shape().size());
        std::copy(seed.shape().begin(), seed.shape().end(), shape.begin());
        auto idim = r.dims.rbegin();
        ssize_t offset = r.dims.size();
        for(ssize_t ishape = std::ssize(shape) - 1; ishape >= 0; ishape--)
        {
            if(offset != 0 && ishape == *idim)
            {
                shape[ishape] = 1;
                idim++;
                offset -= 1;
            }
            else
            {
                shape[ishape] = shape[ishape - offset];
            }
        }
        seed = seed.reshape(std::move(shape));
    }

    switch(r.type)
    {
    case ReduceOpType::SUM:
        // ∇(sum(x)) = { s * ∂x }
        Differentiate(ctx, r.x, seed);
        break;
    case ReduceOpType::MAX:
        // ∇(max(x)) = { s * (x == max(x)) * ∂x }
        Differentiate(ctx, r.x, seed * (r.x == node));
        break;
    }
}

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, const ViewOp &v, GraphNodeHandle seed)
{
    // Don't bother with anything if we reshaped an Immediate
    if(v.x.shape().empty())
        return;
    // Fast path for if we do a squeeze/unsqueeze-type operation
    if(v.x->NumElements() == seed->NumElements())
    {
        auto inverse_offset = -v.offset;
        seed = seed.as_strided(v.x.shape(), v.x.strides(), inverse_offset);
        return Differentiate(ctx, v.x, seed);
    }

    if(seed->NumElements() > node->NumElements())
    {
        dim_t ratio = seed->NumElements() / node->NumElements();
        Shape shape = { ratio };
        shape.insert(shape.end(), v.x.shape().begin(), v.x.shape().end());
        seed = seed.reshape(std::move(shape));
        seed = seed.sum(dim_t{0});
    }

    if(std::find(v.strides.begin(), v.strides.end(), 0) != v.strides.end())
    {
        Dims sum_dims;
        for(dim_t idim = 0; idim < std::ssize(v.strides); idim++)
            if(v.strides[idim] == 0)
                sum_dims.push_back(idim);
        
        seed = seed.sum(std::move(sum_dims));
    }
    if(seed->NumElements() != v.x->NumElements())
        throw std::runtime_error("Differentiating general as_strided is currently unsupported");
    auto inverse_view = seed.as_strided(v.x.shape(), v.x.strides(), 0);
    // TODO: Make this support general as_strided. See https://github.com/pytorch/pytorch/blob/ffc7158bf2f97916305217e4203ef846c00161ce/tools/autograd/templates/Functions.cpp#L937-L1488
    // For now, we only support the case where the node can be broadcasted to the seed.
    Differentiate(ctx, v.x, inverse_view);
}

void Differentiate(BackpropContext &ctx, GraphNodeHandle node, GraphNodeHandle seed)
{
    if(!node->needs_gradient)
        return;

    node->Visit([&](auto &&x) { Differentiate(ctx, node, x, seed); });
}

namespace gigagrad
{

TrainingContext CompileTrainingGraph(
    nn::Module &network,
    GraphNodeHandle loss,
    std::unique_ptr<codegen::Backend> backend,
    float learning_rate)
{
    GraphNodeHandle seed = network.Immediate(learning_rate);
    BackpropContext ctx;
    Differentiate(ctx, loss, seed);
    CodegenNode(ctx.program, loss);
    size_t loss_buffer_id = ctx.program.buffers.size() - 1;

    std::unordered_map<size_t, size_t> weights_to_buffers;
    for(size_t weight : network.weights)
    {
        size_t node_idx = network.graph.inputs[weight];
        weights_to_buffers[node_idx] = -1;
    }
    for(size_t ibuffer = 0; ibuffer < ctx.program.buffers.size(); ibuffer++)
    {
        const codegen::BufferDescriptor &id = ctx.program.buffers[ibuffer];
        if(std::holds_alternative<GraphNodeHandle>(id.id))
        {
            GraphNodeHandle tensor = std::get<GraphNodeHandle>(id.id);
            if(weights_to_buffers.contains(tensor.node_idx))
                weights_to_buffers[tensor.node_idx] = ibuffer;
        }
    }

    for(size_t i = 0; i < ctx.gradients.size(); i++)
    {
        size_t weight_idx = ctx.gradients[i].input.node_idx;
        if(weights_to_buffers.contains(weight_idx))
            CodegenNode(ctx.program, ctx.gradients[i].gradient);
    }

    for(size_t i = 0; i < ctx.gradients.size(); i++)
    {
        size_t weight_idx = ctx.gradients[i].input.node_idx;
        if(weights_to_buffers.contains(weight_idx))
        {
            size_t weight_buffer_idx = weights_to_buffers[weight_idx];
            auto [weight, gradient] = ctx.gradients[i];
            CodegenNode(ctx.program, (weight - gradient), weight_buffer_idx);
        }
    }
    backend->LowerProgram(std::move(ctx.program));
    backend->InitBuffers();

    float *loss_buffer = static_cast<float *>(backend->GetBuffer(loss_buffer_id));
    return { loss_buffer, std::move(backend) };
}
}
