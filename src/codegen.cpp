#include "graph.h"
#include "codegen.h"
#include "backend.h"

#include <cstdio>

namespace gigagrad
{
namespace codegen
{
static Shape ComputeStrides(Shape shape)
{
    dim_t cur = 1;
    for(ssize_t i = std::ssize(shape) - 1; i >= 0; i--)
    {
        auto tmp = shape[i];
        shape[i] = cur;
        cur *= tmp;
    }
    return shape;
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, const GraphNode &node, size_t load_idx);

size_t CodegenNode(Program &prog, FunctionBuilder &f, const Tensor &t, size_t load_idx)
{
    size_t buffer_id = prog.AddBuffer(t);
    auto input = f.Input(buffer_id);
    return f.Load(input, load_idx);
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, const Immediate &i, size_t load_idx)
{
    return f.Immediate(i.value);
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, const UnaryOp &u, size_t load_idx)
{
    auto x = CodegenNode(prog, f, u.x, load_idx);
    return f.Unary(u.type, x);
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, const BinaryOp &u, size_t load_idx)
{
    Shape xshape = u.x.shape();
    Shape xstrides = ComputeStrides(xshape);
    Shape yshape = u.y.shape();
    Shape ystrides = ComputeStrides(yshape);
    Shape broadcasted_shape = GraphNode{u}.shape();

    auto generate_stride_adjustments =
        [&f, &broadcasted_shape, load_idx](const Shape &shape, const Shape &strides)
        {
            auto load = load_idx;
            for(ssize_t i = std::ssize(shape) - 1; i >= 0; i--)
            {
                if(broadcasted_shape[i] != 1 && shape[i] == 1)
                {
                    auto stride = f.IntImmediate(strides[i]);
                    auto broadcasted = f.IntImmediate(broadcasted_shape[i]);
                    auto div = f.Arithmetic(load, IntArithmeticInsn::Op::DIV, stride);
                    auto mod = f.Arithmetic(div, IntArithmeticInsn::Op::MOD, broadcasted);
                    auto mul = f.Arithmetic(mod, IntArithmeticInsn::Op::MUL, stride);
                    load = f.Arithmetic(load, IntArithmeticInsn::Op::SUB, mul);
                }
            }
            return load;
        };

    size_t xload = generate_stride_adjustments(xshape, xstrides);
    size_t yload = generate_stride_adjustments(yshape, ystrides);
    auto x = CodegenNode(prog, f, u.x, xload);
    auto y = CodegenNode(prog, f, u.y, yload);
    return f.Binary(u.type, x, y);
}

size_t CodegenNode(Program &prog, FunctionBuilder &old_f, const ReduceOp &r, size_t output_load_idx)
{
    FunctionBuilder f(GraphNode{r}.shape());

    Shape shape = r.x.shape();
    Shape strides = ComputeStrides(shape);

    auto FixDim = [&](dim_t dim)
    {
        auto mod = static_cast<dim_t>(shape.size());
        return ((dim % mod) + mod) % mod;
    };

    Shape output_shape = shape;
    for(auto dim : r.dims)
        output_shape[FixDim(dim)] = 1;
    Shape output_strides = ComputeStrides(output_shape);

    std::vector<size_t> accumulators;
    auto reduce_dim = r.dims.begin(); // dims is sorted
    auto store_idx = f.IntImmediate(0);
    auto load_idx = store_idx;

    // Generate loops for all of the non-reducing dimensions
    for(ssize_t i = 0; i < std::ssize(shape); i++)
    {
        if(reduce_dim == r.dims.end() || i != FixDim(*reduce_dim))
        {
            auto loop = f.Loop(shape[i], strides[i]);
            auto input_stride = f.IntImmediate(strides[i]);
            auto output_stride = f.IntImmediate(output_strides[i]);
            auto mul_input_stride = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, input_stride);
            auto mul_output_stride = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, output_stride);
            load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul_input_stride);
            store_idx = f.Arithmetic(store_idx, IntArithmeticInsn::Op::ADD, mul_output_stride);
        }
        else if(i == FixDim(*reduce_dim))
            *reduce_dim++;
    }
    // Generate loops along reduction dimension
    for(auto dim : r.dims)
    {
        auto mod = static_cast<dim_t>(shape.size());
        auto fixed_dim = ((dim % mod) + mod) % mod;
        accumulators.push_back(f.Immediate(0.0f));
        auto loop = f.Loop(shape[fixed_dim], strides[fixed_dim]);
        auto stride = f.IntImmediate(strides[fixed_dim]);
        auto mul = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, stride);
        load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul);
    }

    auto to_accumulate = CodegenNode(prog, f, r.x, load_idx);

    ssize_t iaccum = std::ssize(r.dims) - 1;
    do
    {
        f.Accumulate(r.type, accumulators[iaccum], to_accumulate);
        to_accumulate = accumulators[iaccum];
        f.EndLoop();
        iaccum--;
    } while(iaccum >= 0);
    f.Store(store_idx, accumulators[0]);
    for(ssize_t i = 0; i < std::ssize(shape) - std::ssize(r.dims); i++)
        f.EndLoop();

    prog.PushFunction(std::move(f));
    auto input = old_f.Input(prog.functions.back().output_buffer);
    return old_f.Load(input, output_load_idx);
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, const ViewOp &p, size_t load_idx)
{
    auto x = CodegenNode(prog, f, p.x, load_idx);
    return x;
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, const GraphNode &node, size_t load_idx)
{
    return std::visit([&](auto &&x) { return CodegenNode(prog, f, x, load_idx); }, node);
}

void EnterCodegen(Program &prog, const GraphNode &node)
{
    // ReduceOp generates its own for loops
    if(std::holds_alternative<gigagrad::ReduceOp>(node))
    {
        FunctionBuilder f(node.shape());
        CodegenNode(prog, f, node, 0);
    }
    else
    {
        FunctionBuilder f(node.shape());
        Shape shape = node.shape();
        Shape strides = ComputeStrides(shape);
        auto load_idx = f.IntImmediate(0);
        for(ssize_t i = 0; i < std::ssize(shape); i++)
        {
            auto loop = f.Loop(shape[i], strides[i]);
            auto stride = f.IntImmediate(strides[i]);
            auto mul = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, stride);
            load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul);
        }
        auto to_store = CodegenNode(prog, f, node, load_idx);
        f.Store(load_idx, to_store);
        for(ssize_t i = 0; i < std::ssize(shape); i++)
            f.EndLoop();
        prog.PushFunction(std::move(f));
    }
}
}

codegen::Program CodegenNode(const GraphNode &node)
{
    codegen::Program result;
    codegen::EnterCodegen(result, node);
    return result;
}

CompiledTensor GraphNode::Compile(std::unique_ptr<codegen::Backend> backend) const
{
    codegen::Program prog = CodegenNode(*this);
    backend->LowerProgram(std::move(prog));

    CompiledTensor result;
    result.shape = this->shape();
    result.data = reinterpret_cast<float *>(backend->InitBuffers());
    result.backend = std::move(backend);

    return result;
}

}
