#pragma once

#ifdef HAS_METAL

#include "backend.h"
#include "codegen.h"

namespace MTL
{
class Buffer;
class Device;
class Library;
class CommandQueue;
class CommandBuffer;
class ComputePipelineState;
class ComputeCommandEncoder;
}

namespace gigagrad
{
namespace codegen
{

struct BackendMetal : public Backend
{
    using GraphEvalFn = void (*)(void **);
    virtual ~BackendMetal();
    virtual void LowerProgram(Program &&program);
    virtual void *InitBuffers();
    virtual void *GetBuffer(size_t idx);
    virtual void Execute();

    MTL::Buffer* Allocate(size_t size);
    void LoadLibrary();
    void RunKernel(std::string kernel_name);

    size_t outer_dim;
    MTL::Device *device;
    MTL::Library *library;
    MTL::CommandQueue *queue;
    MTL::CommandBuffer *command_buffer;
    MTL::ComputeCommandEncoder *enc;
    MTL::ComputePipelineState *pipeline;
    void *handle;
    Program program;
    std::vector<MTL::Buffer *> buffers;
    GraphEvalFn eval_fn;
};


}
}

#endif
