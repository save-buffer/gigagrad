#include "graph.h"
#include "codegen.h"
#include "backend.h"

#include <algorithm>
#include <cstdio>

namespace gigagrad
{
namespace codegen
{

size_t CodegenNode(Program &prog, FunctionBuilder &f, GraphNodeHandle node, size_t load_idx, size_t max_seen_size_elts);

size_t CodegenNode(
    Program &prog,
    FunctionBuilder &f,
    GraphNodeHandle node,
    const Tensor &t,
    size_t load_idx,
    size_t max_seen_size_elts)
{
    size_t size_elts = std::accumulate(node.shape().begin(), node.shape().end(), 1, std::multiplies{});
    size_elts = std::max(max_seen_size_elts, size_elts);
    size_t buffer_id = prog.AddBuffer(node, size_elts);
    auto input = f.Input(buffer_id);
    return f.Load(input, load_idx);
}

size_t CodegenNode(
    Program &prog,
    FunctionBuilder &f,
    GraphNodeHandle,
    const Immediate &i,
    size_t load_idx,
    size_t max_seen_size_elts)
{
    return f.Immediate(i.value);
}

size_t CodegenNode(
    Program &prog,
    FunctionBuilder &f,
    GraphNodeHandle,
    const UnaryOp &u,
    size_t load_idx,
    size_t max_seen_size_elts)
{
    auto x = CodegenNode(prog, f, u.x, load_idx, max_seen_size_elts);
    return f.Unary(u.type, x);
}

size_t CodegenNode(
    Program &prog,
    FunctionBuilder &f,
    GraphNodeHandle node,
    const BinaryOp &b,
    size_t load_idx,
    size_t max_seen_size_elts)
{
    auto x = CodegenNode(prog, f, b.x, load_idx, max_seen_size_elts);
    auto y = CodegenNode(prog, f, b.y, load_idx, max_seen_size_elts);
    return f.Binary(b.type, x, y);
}

size_t CodegenNode(
    Program &prog,
    FunctionBuilder &old_f,
    GraphNodeHandle node,
    const ReduceOp &r,
    size_t output_load_idx,
    size_t max_seen_size_elts)
{
    FunctionBuilder f(node, max_seen_size_elts);

    std::vector<size_t> accumulators;
    auto reduce_dim = r.dims.begin(); // dims is sorted
    auto ioutput_strides = node.strides().begin();

    auto store_idx = f.IntImmediate(0);
    auto load_idx = store_idx;

    const Shape &input_shape = r.x.shape();
    const Shape &input_strides = r.x.strides();

    // Generate loops for all of the non-reducing dimensions
    for(ssize_t i = 0; i < std::ssize(input_shape); i++)
    {
        if(reduce_dim == r.dims.end() || i != *reduce_dim)
        {
            auto loop = f.Loop(input_shape[i], input_strides[i]);
            auto input_stride = f.IntImmediate(input_strides[i]);
            auto output_stride = f.IntImmediate(*ioutput_strides);
            auto mul_input_stride = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, input_stride);
            auto mul_output_stride = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, output_stride);
            load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul_input_stride);
            store_idx = f.Arithmetic(store_idx, IntArithmeticInsn::Op::ADD, mul_output_stride);

            if(!r.keepdim)
                ioutput_strides++;
        }
        else if(i == *reduce_dim)
        {
            reduce_dim++;
        }

        // If keepdim, always advance output_strides because number of input/output
        // dimensions matches
        if(r.keepdim)
            ioutput_strides++;
    }
    // Generate loops along reduction dimension
    for(auto dim : r.dims)
    {
        accumulators.push_back(f.Immediate(0.0f));
        auto loop = f.Loop(input_shape[dim], input_strides[dim]);
        auto stride = f.IntImmediate(input_strides[dim]);
        auto mul = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, stride);
        load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul);
    }

    auto to_accumulate = CodegenNode(prog, f, r.x, load_idx, 0);

    ssize_t iaccum = std::ssize(r.dims) - 1;
    do
    {
        f.Accumulate(r.type, accumulators[iaccum], to_accumulate);
        to_accumulate = accumulators[iaccum];
        f.EndLoop();
        iaccum--;
    } while(iaccum >= 0);
    f.Store(store_idx, accumulators[0]);
    for(ssize_t i = 0; i < std::ssize(input_shape) - std::ssize(r.dims); i++)
        f.EndLoop();

    prog.PushFunction(std::move(f));
    auto input = old_f.Input(prog.functions.back().output_buffer);
    return old_f.Load(input, output_load_idx);
}

size_t CodegenNode(
    Program &prog,
    FunctionBuilder &f,
    GraphNodeHandle node,
    const ViewOp &v,
    size_t load_idx,
    size_t max_seen_size)
{
    // TODO: This generates a lot of unnecessary crap right now, ideally this arithmetic
    //       expression would get accumulated and simplified before being emitted.
    const Shape &shape = v.shape;
    const Shape &strides = v.strides;
    const Shape &output_strides = node.strides();

    auto new_load_idx = f.IntImmediate(0);
    for(ssize_t i = std::ssize(shape) - 1; i >= 0; i--)
    {
        auto output_stride = f.IntImmediate(output_strides[i]);
        auto output_shape = f.IntImmediate(shape[i]);
        auto input_stride = f.IntImmediate(strides[i]);
        auto div = f.Arithmetic(load_idx, IntArithmeticInsn::Op::DIV, output_stride);
        auto mod = f.Arithmetic(div, IntArithmeticInsn::Op::MOD, output_shape);
        auto mul = f.Arithmetic(mod, IntArithmeticInsn::Op::MUL, input_stride);
        new_load_idx = f.Arithmetic(new_load_idx, IntArithmeticInsn::Op::ADD, mul);
    }
    auto offset = f.IntImmediate(v.offset);
    new_load_idx = f.Arithmetic(new_load_idx, IntArithmeticInsn::Op::ADD, offset);
    size_t view_size = std::accumulate(
        v.shape.begin(),
        v.shape.end(),
        dim_t{1},
        std::multiplies{});
    view_size -= v.offset;

    max_seen_size = std::max(max_seen_size, view_size);
    auto x = CodegenNode(prog, f, v.x, new_load_idx, max_seen_size);
    return x;
}

size_t CodegenNode(Program &prog, FunctionBuilder &f, GraphNodeHandle node, size_t load_idx, size_t max_seen_size_elts)
{
    if(prog.node_function_cache.contains(node.node_idx))
    {
        size_t function_id = prog.node_function_cache[node.node_idx];
        size_t buffer_id = prog.functions[function_id].output_buffer;
        prog.buffers[buffer_id].size_elts = std::max(prog.buffers[buffer_id].size_elts, max_seen_size_elts);
        auto input = f.Input(buffer_id);
        return f.Load(input, load_idx);
    }
    return node->Visit([&](auto &&x)
    {
        return CodegenNode(prog, f, node, x, load_idx, max_seen_size_elts);
    });
}

void CodegenNode(Program &prog, GraphNodeHandle node, std::optional<size_t> output_buffer)
{
    // ReduceOp generates its own loops
    if(node->Kind() == GraphNode::Kind::ReduceOp)
    {
        FunctionBuilder f(node);
        CodegenNode(prog, f, node, 0, 0);
    }
    else
    {
        FunctionBuilder f(node);
        const Shape &shape = node.shape();
        const Shape &strides = node.strides();
        auto load_idx = f.IntImmediate(0);
        for(ssize_t i = 0; i < std::ssize(shape); i++)
        {
            auto loop = f.Loop(shape[i], strides[i]);
            auto stride = f.IntImmediate(strides[i]);
            auto mul = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, stride);
            load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul);
        }
        auto to_store = CodegenNode(prog, f, node, load_idx, 0);
        f.Store(load_idx, to_store);
        for(ssize_t i = 0; i < std::ssize(shape); i++)
            f.EndLoop();

        prog.PushFunction(std::move(f));
    }
    if(output_buffer)
    {
        // This can potentially be a little fragile, but it's simple and easy for now.
        // We rely on the fact that PushFunction always adds a new buffer, so if we want
        // to remap the last function's output to something else, we can just pop_back().
        prog.buffers.pop_back();
        prog.ChangeOutputBuffer(prog.functions.size() - 1, *output_buffer);
    }
}

codegen::Program CodegenNode(GraphNodeHandle node)
{
    codegen::Program result;
    codegen::CodegenNode(result, node);
    return result;
}

}

CompiledTensor GraphNodeHandle::Compile(std::unique_ptr<codegen::Backend> backend) const
{
    codegen::Program prog = codegen::CodegenNode(*this);
    backend->LowerProgram(std::move(prog));

    CompiledTensor result;
    result.shape = this->shape();
    result.data = reinterpret_cast<float *>(backend->InitBuffers());
    result.backend = std::move(backend);

    return result;
}

}
