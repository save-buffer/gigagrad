#include "runtime.h"

using namespace Gigagrad;
using namespace Gigagrad::Codegen;

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

namespace Gigagrad
{

void Eval(Gigagrad::Codegen::GraphEvalFn fn, std::vector<BufferDescriptor> buffer_descs)
{
    std::vector<void *> buffers = AllocateBuffers(buffer_descs);
    for(size_t ibuff = 0; ibuff < buffer_descs.size(); ibuff++)
    {
        if(std::holds_alternative<const Tensor *>(buffer_descs[ibuff].id))
        {
            const Tensor *tensor = std::get<const Tensor *>(buffer_descs[ibuff].id);
            tensor->load_callback(buffers[ibuff])
        }
    }
    fn(buffers.data());
}

}
