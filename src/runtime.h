#pragma once
#include "backend.h"

namespace Gigagrad
{

void Eval(Codegen::GraphEvalFn fn, std::vector<Codegen::BufferDescriptor> buffer_descs);

}
