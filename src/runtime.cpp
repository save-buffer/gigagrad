#include "runtime.h"

using namespace gigagrad;
using namespace gigagrad::codegen;

static std::vector<void *> AllocateBuffers(const std::vector<BufferDescriptor> &buffer_descs)
{
    std::vector<void *> result(buffer_descs.size());
    for(size_t i = 0; i < buffer_descs.size(); i++)
    {
        result[i] = malloc(sizeof(float) * buffer_descs[i].size_elts);
        if(!result[i])
            throw std::runtime_error("Failed to allocate buffer");
    }
    return result;
}

namespace gigagrad
{

void Eval(gigagrad::codegen::GraphEvalFn fn, std::vector<BufferDescriptor> buffer_descs)
{
    std::vector<void *> buffers = AllocateBuffers(buffer_descs);
    for(size_t ibuff = 0; ibuff < buffer_descs.size(); ibuff++)
    {
        if(std::holds_alternative<GraphNodeHandle>(buffer_descs[ibuff].id))
        {
            GraphNodeHandle tensor = std::get<GraphNodeHandle>(buffer_descs[ibuff].id);
            tensor->u.t.tensor.load_callback(buffers[ibuff]);
        }
    }
    fn(buffers.data());
}

}
