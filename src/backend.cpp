#include "backend.h"
#include "backend_scalar_c.h"
#include "codegen.h"

namespace Gigagrad
{
namespace Codegen
{

GraphEvalFn LowerProgram(const char *prefix, Backend backend, const Program &program)
{
    switch(backend)
    {
    case Backend::ScalarC:
        return Internal::Lower_ScalarC(prefix, program);
    default:
        throw std::domain_error("Invalid backend");
    }
}
}
}
