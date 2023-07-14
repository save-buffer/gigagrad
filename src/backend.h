#pragma once
#include "codegen.h"

namespace Gigagrad
{
namespace Codegen
{

enum class Backend
{
    ScalarC,
};

void LowerProgram(const char *prefix, Backend backend, const Program &program);

}
}
