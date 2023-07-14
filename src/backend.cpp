#include "backend.h"
#include "codegen.h"

namespace Gigagrad
{
namespace Codegen
{

void Lower_ScalarC(const LoadIntImmediateInsn &i, size_t iinsn, int &indentation)
{
    std::printf("%*sint64_t v%zu = %lld;\n", indentation, " ", iinsn, (long long)i.value);
}

void Lower_ScalarC(const IntArithmeticInsn &i, size_t iinsn, int &indentation)
{
    std::printf("%*sint64_t v%zu = v%zu %c v%zu;\n", indentation, " ", iinsn, i.x, (char)i.op, i.y);
}

void Lower_ScalarC(const BeginLoopInsn &i, size_t iinsn, int &indentation)
{
    std::printf("%*sfor(int64_t v%zu = 0; v%zu < %zd; v%zu++)\n%*s{\n",
                indentation, " ", iinsn, iinsn, i.range, iinsn, indentation, " ");
    indentation += 4;
}

void Lower_ScalarC(const EndLoopInsn &i, size_t iinsn, int &indentation)
{
    indentation -= 4;
    std::printf("%*s}\n", indentation, " ");
}

void Lower_ScalarC(const LoadInsn &i, size_t iinsn, int &indentation)
{
    std::printf("%*sfloat v%zu = i%zu[v%zu];\n",
                indentation, " ", iinsn, i.input, i.idx);
}

void Lower_ScalarC(const StoreInsn &i, size_t iinsn, int &indentation)
{
    std::printf("%*soutput[v%zu] = v%zu;\n", indentation, " ", i.offset, i.value);
}

void Lower_ScalarC(const LoadImmediateInsn &i, size_t iinsn, int &indentation)
{
    std::printf("%*sfloat v%zu = %f;\n", indentation, " ", iinsn, i.value);
}

void Lower_ScalarC(const UnaryInsn &i, size_t iinsn, int &indentation)
{
    auto op_str = i.type == UnaryOpType::EXP ? "exp"
        : i.type == UnaryOpType::LOG ? "log"
        : i.type == UnaryOpType::SIN ? "sin"
        : "INVALID";
    std::printf("%*sfloat v%zu = %s(v%zu);\n",
                indentation, " ", iinsn, op_str, i.x);
}

void Lower_ScalarC(const BinaryInsn &i, size_t iinsn, int &indentation)
{
    if(i.type == BinaryOpType::ADD
       || i.type == BinaryOpType::SUB
       || i.type == BinaryOpType::MUL
       || i.type == BinaryOpType::DIV
       || i.type == BinaryOpType::CMP)
    {
        auto op_str = i.type == BinaryOpType::ADD ? "+"
            : i.type == BinaryOpType::SUB ? "-"
            : i.type == BinaryOpType::MUL ? "*"
            : i.type == BinaryOpType::DIV ? "/"
            : "==";

        std::printf("%*sfloat v%zu = (float)(v%zu %s v%zu);\n",
                    indentation, " ", iinsn, i.x, op_str, i.y);
    }
    else if(i.type == BinaryOpType::MAX)
    {
        std::printf("%*sfloat v%zu = v%zu > v%zu ? v%zu : v%zu;\n",
                    indentation, " ", iinsn, i.x, i.y, i.x, i.y);
    }
    else
    {
        std::printf("%*sfloat v%zu = pow(v%zu, v%zu);\n",
                    indentation, " ", iinsn, i.x, i.y);
    }
}

void Lower_ScalarC(const AccumulateInsn &i, size_t iinsn, int &indentation)
{
    if(i.type == ReduceOpType::MAX)
        std::printf("%*sv%zu = v%zu > v%zu ? v%zu : v%zu;\n",
                    indentation, " ", i.accumulator, i.accumulator, i.x, i.accumulator, i.x);
    else
        std::printf("%*sv%zu += v%zu;\n", indentation, " ", i.accumulator, i.x);
}

void Lower_ScalarC(const char *prefix, size_t ifn, const FunctionBuilder &fn)
{
    std::printf("void %s_%zu(\n", prefix, ifn);
    for(size_t i = 0; i < fn.inputs.size(); i++)
        std::printf("    const float *i%zu,\n", i);
    std::printf("    float *output)\n{\n");
    int indentation = 4;
    for(size_t i = 0; i < fn.insns.size(); i++)
    {
        std::visit([&](auto &&insn) { Lower_ScalarC(insn, i, indentation); }, fn.insns[i]);
    }
    std::printf("}\n");
}

void Lower_ScalarC(const char *prefix, const Program &program)
{
    for(size_t ifn = 0; ifn < program.functions.size(); ifn++)
        Lower_ScalarC(prefix, ifn, program.functions[ifn]);
}

void LowerProgram(const char *prefix, Backend backend, const Program &program)
{
    switch(backend)
    {
    case Backend::ScalarC:
        return Lower_ScalarC(prefix, program);
    default:
        throw std::domain_error("Invalid backend");
    }
}
}
}
