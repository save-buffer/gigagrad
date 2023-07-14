#include "graph.h"
#include "codegen.h"
#include "backend.h"

#include <cstdio>

namespace Gigagrad
{
namespace Codegen
{
Shape ComputeStrides(Shape shape)
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

size_t CodegenNode(Program &prog, const GraphNode &node, size_t load_idx);

size_t CodegenNode(Program &prog, const Tensor &t, size_t load_idx)
{
    FunctionBuilder &f = prog.CurrentFunction();
    auto input = f.Input(t);
    return f.Load(input, load_idx);
}

size_t CodegenNode(Program &prog, const Immediate &i, size_t load_idx)
{
    FunctionBuilder &f = prog.CurrentFunction();
    return f.Immediate(i.value);
}

size_t CodegenNode(Program &prog, const UnaryOp &u, size_t load_idx)
{
    FunctionBuilder &f = prog.CurrentFunction();
    auto x = CodegenNode(prog, u.x, load_idx);
    return f.Unary(u.type, x);
}

size_t CodegenNode(Program &prog, const BinaryOp &u, size_t load_idx)
{
    FunctionBuilder &f = prog.CurrentFunction();
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
    auto x = CodegenNode(prog, u.x, xload);
    auto y = CodegenNode(prog, u.y, yload);
    return f.Binary(u.type, x, y);
}

size_t CodegenNode(Program &prog, const ReduceOp &r, size_t output_load_idx)
{
    FunctionBuilder &f = prog.PushFunction();
    Shape shape = r.x.shape();
    Shape strides = ComputeStrides(shape);

    std::vector<size_t> accumulators;
    auto reduce_dim = r.dims.begin();
    auto store_idx = f.IntImmediate(0);
    for(ssize_t i = 0; i < std::ssize(shape); i++)
    {
        if(reduce_dim == r.dims.end() || i != *reduce_dim)
        {
            auto loop = f.Loop(shape[i], strides[i]);
            auto stride = f.IntImmediate(strides[i]);
            auto mul = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, stride);
            store_idx = f.Arithmetic(store_idx, IntArithmeticInsn::Op::ADD, mul);
        }
    }
    auto load_idx = store_idx;
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

    auto to_accumulate = CodegenNode(prog, r.x, load_idx);

    ssize_t iaccum = std::ssize(r.dims) - 1;
    do
    {
        f.Accumulate(r.type, accumulators[iaccum], to_accumulate);
        to_accumulate = accumulators[iaccum];
        f.EndLoop();
        iaccum--;
    } while(iaccum >= 0);
    f.Store(store_idx, accumulators[0]);
    for(ssize_t i = 0; i < std::ssize(shape) - std::ssize(r.dims) + 1; i++)
        f.EndLoop();

    prog.PopFunction();

    if(prog.HasIncompleteFunctions())
    {
        FunctionBuilder &f = prog.CurrentFunction();
        auto input = f.Input(r);
        return f.Load(input, output_load_idx);
    }
    return 0;
}

size_t CodegenNode(Program &prog, const ReshapeOp &r, size_t load_idx)
{
    auto x = CodegenNode(prog, r.x, load_idx);
    return x;
}

size_t CodegenNode(Program &prog, const PermuteOp &p, size_t load_idx)
{
    auto x = CodegenNode(prog, p.x, load_idx);
    return x;
}

size_t CodegenNode(Program &prog, const GraphNode &node, size_t load_idx)
{
    return std::visit([&](auto &&x) { return CodegenNode(prog, x, load_idx); }, node);
}

void EnterCodegen(Program &prog, const GraphNode &node)
{
    // ReduceOp generates its own for loops
    if(std::holds_alternative<Gigagrad::ReduceOp>(node))
    {
        CodegenNode(prog, node, 0);
        return;
    }

    FunctionBuilder &f = prog.PushFunction();
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
    auto to_store = CodegenNode(prog, node, load_idx);
    f.Store(load_idx, to_store);
    for(ssize_t i = 0; i < std::ssize(shape); i++)
        f.EndLoop();
    prog.PopFunction();
}
}

void PrintCodegenNode(GraphNode &node)
{
    Codegen::Program prog;
    EnterCodegen(prog, node);
    prog.Print();
    LowerProgram("BORK", Codegen::Backend::ScalarC, prog);
}

Codegen::Program CodegenNode(GraphNode &node)
{
    Codegen::Program result;
    Codegen::EnterCodegen(result, node);
    return result;
}

}
