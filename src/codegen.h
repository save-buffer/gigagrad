#pragma once

#include <cstdint>
#include <cstdio>

#include "graph.h"

namespace Gigagrad
{
namespace Codegen
{
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
    void Print(size_t)
    {
        std::printf("END LOOP\n");
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
    size_t offset;
    size_t value;

    void Print(size_t iinsn)
    {
        std::printf("Output[v%zu] = v%zu\n", offset, value);
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

struct AccumulateInsn
{
    ReduceOpType type;
    size_t accumulator;
    size_t x;

    void Print(size_t iinsn)
    {
        auto op_str = type == ReduceOpType::MAX ? "MAX"
            : type == ReduceOpType::SUM ? "SUM"
            : "INVALID";
        std::printf("v%zu <- %s(v%zu, v%zu)\n", accumulator, op_str, accumulator, x);
    }
};

using Instruction = std::variant<
    LoadIntImmediateInsn,
    IntArithmeticInsn,
    BeginLoopInsn,
    EndLoopInsn,
    LoadInsn,
    StoreInsn,
    LoadImmediateInsn,
    UnaryInsn,
    BinaryInsn,
    AccumulateInsn>;

struct FunctionBuilder
{
    size_t Loop(dim_t range, dim_t stride)
    {
        insns.emplace_back(BeginLoopInsn{range, stride});
        return insns.size() - 1;
    }

    size_t EndLoop()
    {
        insns.emplace_back(EndLoopInsn{});
        return insns.size() - 1;
    }

    size_t Input(const GraphNode &g)
    {
        inputs.push_back(&g);
        return inputs.size() - 1;
    }

    size_t Input(const size_t fn_idx)
    {
        inputs.push_back(fn_idx);
        return inputs.size() - 1;
    }

    size_t Load(size_t input_idx, size_t load_idx)
    {
        insns.emplace_back(LoadInsn{input_idx, load_idx});
        return insns.size() - 1;
    }

    size_t Store(size_t offset, size_t value)
    {
        insns.emplace_back(StoreInsn{offset, value});
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

    size_t Accumulate(ReduceOpType type, size_t accumulator, size_t x)
    {
        insns.emplace_back(AccumulateInsn{type, accumulator, x});
        return insns.size() - 1;
    }

    void Print()
    {
        for(ssize_t i = 0; i < std::ssize(insns); i++)
        {
            std::visit([&](auto &&insn) { insn.Print(i); }, insns[i]);
        }
    }

    std::vector<Instruction> insns;
    std::vector<std::variant<const GraphNode *, size_t>> inputs;
};

struct Program
{
    void PushFunction(FunctionBuilder function)
    {
        functions.emplace_back(std::move(function));
    }

    size_t NumFunctions()
    {
        return functions.size();
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
}
void PrintCodegenNode(GraphNode &node);
Codegen::Program CodegenNode(GraphNode &node);
}
