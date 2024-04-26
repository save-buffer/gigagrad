#pragma once
#include <cstddef>

namespace gigagrad
{
namespace codegen
{

struct Program;

struct Backend
{
    virtual ~Backend() = default;
    virtual void LowerProgram(Program &&program) = 0;
    virtual void *InitBuffers() = 0; // Returns output buffer
    virtual void *GetBuffer(size_t idx) = 0;
    virtual void Execute() = 0;
};

}
}
