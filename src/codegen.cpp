#include "graph.h"

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

struct LoadIntImmediateInsn
{
    int64_t value;

    void Print(size_t iinsn)
    {
        std::printf("v%zu = %lld\n", iinsn, (long long)value);
    }
};

struct IntArithmeticInsn
{
    enum class Op : char
    {
        ADD = '+',
        SUB = '-',
        MUL = '*',
        DIV = '/',
        MOD = '%',
    };
    Op op;
    size_t x;
    size_t y;

    void Print(size_t iinsn)
    {
        std::printf("v%zu = v%zu %c v%zu\n", iinsn, x, (char)op, y);
    }
};

struct BeginLoopInsn
{
    dim_t range;
    dim_t stride;

    void Print(size_t iinsn)
    {
        std::printf("v%zu = LOOP [0..%zd, %zd]\n", iinsn, range, stride);
    }
};

struct EndLoopInsn
{
    size_t loop;

    void Print(size_t iinsn)
    {
        std::printf("END LOOP v%zu\n", loop);
    }
};

struct LoadInsn
{
    size_t input;
    size_t idx;

    void Print(size_t iinsn)
    {
        std::printf("v%zu = LOAD I%zu[v%zu]\n", iinsn, input, idx);
    }
};

struct StoreInsn
{
    size_t idx;

    void Print(size_t iinsn)
    {
        std::printf("STORE [v%zu]\n", idx);
    }
};

struct LoadImmediateInsn
{
    float value;

    void Print(size_t iinsn)
    {
        std::printf("v%zu = %f\n", iinsn, value);
    }
};

struct UnaryInsn
{
    UnaryOpType type;
    size_t x;

    void Print(size_t iinsn)
    {
        auto op_str = type == UnaryOpType::NOP ? "NOP"
            : type == UnaryOpType::EXP ? "EXP"
            : type == UnaryOpType::LOG ? "LOG"
            : type == UnaryOpType::CAST ? "CAST"
            : type == UnaryOpType::SIN ? "SIN"
            : "INVALID";
        std::printf("v%zu = %s(v%zu)\n", iinsn, op_str, x);
    }
};

struct BinaryInsn
{
    BinaryOpType type;
    size_t x;
    size_t y;

    void Print(size_t iinsn)
    {
        auto op_str = type == BinaryOpType::ADD ? "+"
            : type == BinaryOpType::SUB ? "-"
            : type == BinaryOpType::MUL ? "*"
            : type == BinaryOpType::DIV ? "/"
            : type == BinaryOpType::POW ? "^"
            : type == BinaryOpType::CMP ? "=="
            : type == BinaryOpType::MAX ? "max"
            : "INVALID";
        std::printf("v%zu = v%zu %s v%zu\n", iinsn, x, op_str, y);
    }
};

using Instruction = std::variant<
    LoadIntImmediateInsn,
    IntArithmeticInsn,
    BeginLoopInsn,
    EndLoopInsn,
    LoadInsn, StoreInsn,
    LoadImmediateInsn,
    UnaryInsn,
    BinaryInsn>;

struct FunctionBuilder
{
    size_t Loop(dim_t range, dim_t stride)
    {
        insns.emplace_back(BeginLoopInsn{range, stride});
        return insns.size() - 1;
    }

    size_t EndLoop(size_t input_insn)
    {
        insns.emplace_back(EndLoopInsn{input_insn});
        return insns.size() - 1;
    }

    size_t Input(const GraphNode &g)
    {
        inputs.push_back(&g);
        return inputs.size() - 1;
    }

    size_t Load(size_t input_idx, size_t load_idx)
    {
        insns.emplace_back(LoadInsn{input_idx, load_idx});
        return insns.size() - 1;
    }

    size_t Store(size_t idx)
    {
        insns.emplace_back(StoreInsn{idx});
        return insns.size() - 1;
    }

    size_t Immediate(float value)
    {
        insns.emplace_back(LoadImmediateInsn{value});
        return insns.size() - 1;
    }

    size_t IntImmediate(int64_t value)
    {
        insns.emplace_back(LoadIntImmediateInsn{value});
        return insns.size() - 1;
    }

    size_t Arithmetic(size_t x, IntArithmeticInsn::Op op, size_t y)
    {
        insns.emplace_back(IntArithmeticInsn{op, x, y});
        return insns.size() - 1;
    }

    size_t Unary(UnaryOpType type, size_t x)
    {
        insns.emplace_back(UnaryInsn{type, x});
        return insns.size() - 1;
    }

    size_t Binary(BinaryOpType type, size_t x, size_t y)
    {
        insns.emplace_back(BinaryInsn{type, x, y});
        return insns.size() - 1;
    }

    void Print()
    {
        for(auto i = 0; i < insns.size(); i++)
        {
            std::visit([&](auto &&insn) { insn.Print(i); }, insns[i]);
        }
    }

    std::vector<Instruction> insns;
    std::vector<const GraphNode *> inputs;
    std::vector<size_t> loops;
    static constexpr size_t Output = static_cast<size_t>(-1);
};

struct Program
{
    FunctionBuilder &NewFunction()
    {
        functions.push_back({});
        return CurrentFunction();
    }

    FunctionBuilder &CurrentFunction()
    {
        return functions.back();
    }

    void Print()
    {
        for(size_t i = 0; i < functions.size(); i++)
        {
            std::printf("BEGIN FUNCTION %zu\n", i);
            functions[i].Print();
            std::printf("END FUNCTION %zu\n", i);
        }
    }

    std::vector<FunctionBuilder> functions;
};

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
                printf("broadcasted_shape: %d, shape: %d\n", broadcasted_shape[i], shape[i]);
                if(broadcasted_shape[i] != 1 && shape[i] == 1)
                {
                    auto stride = f.IntImmediate(strides[i]);
                    auto div = f.Arithmetic(load, IntArithmeticInsn::Op::DIV, stride);
                    auto mul = f.Arithmetic(div, IntArithmeticInsn::Op::MUL, stride);
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

size_t CodegenNode(Program &prog, const ReduceOp &r, size_t load_idx)
{
    FunctionBuilder &f = prog.NewFunction();
    Shape shape = r.x.shape();
    Shape strides = ComputeStrides(shape);
    {
        std::vector<size_t> accumulators;
        auto reduce_dim = r.dims.begin();

        auto load_idx = f.IntImmediate(0);
        for(ssize_t i = 0; i < std::ssize(shape); i++)
        {
            while(i < std::ssize(shape) && (reduce_dim == r.dims.end() || i < *reduce_dim))
            {
                auto loop = f.Loop(shape[i], strides[i]);
                auto stride = f.IntImmediate(strides[i]);
                auto mul = f.Arithmetic(loop, IntArithmeticInsn::Op::MUL, stride);
                load_idx = f.Arithmetic(load_idx, IntArithmeticInsn::Op::ADD, mul);
                i++;
            }
            if(i == std::ssize(shape))
                break;
            reduce_dim++;
            accumulators.push_back(f.Immediate(0.0f));
        }
        CodegenNode(prog, r.x, load_idx);
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

void PrintCodegenNode(GraphNode &node)
{
    Program prog;
    CodegenNode(prog, node, 0);
    prog.Print();
}


void CodegenNode(GraphNode &node)
{
}

}
}
