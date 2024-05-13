#include "backend_metal.h"
#include <filesystem>
#include <system_error>
#include <cstdio>
#include <cerrno>
#include <cstdlib>

#include <dlfcn.h>

using namespace gigagrad;
using namespace gigagrad::codegen;

struct LowerCtx
{
    const char *prefix;
    FILE *file;
    int indentation;
    int loop_depth;
};

static void Lower_Metal(LowerCtx &ctx, const LoadIntImmediateInsn &i, size_t iinsn)
{
    std::fprintf(ctx.file, "%*sint64_t v%zu = %" PRIi64 ";\n", ctx.indentation, " ", iinsn, i.value);
}

static void Lower_Metal(LowerCtx &ctx, const IntArithmeticInsn &i, size_t iinsn)
{
    std::fprintf(ctx.file,
                 "%*sint64_t v%zu = v%zu %c v%zu;\n", ctx.indentation, " ", iinsn, i.x, (char)i.op, i.y);
}

static void Lower_Metal(LowerCtx &ctx, const BeginLoopInsn &i, size_t iinsn)
{
    if(ctx.loop_depth == 0)
    {
        std::fprintf(ctx.file, "%*sint64_t v%zu = outer_idx;\n", ctx.indentation, " ", iinsn);
        ctx.loop_depth += 1;
    }
    else
    {
        std::fprintf(ctx.file, "%*sfor(int64_t v%zu = 0; v%zu < %zd; v%zu++)\n%*s{\n",
                     ctx.indentation, " ", iinsn, iinsn, i.range, iinsn, ctx.indentation, " ");
        ctx.indentation += 4;
    }
}

static void Lower_Metal(LowerCtx &ctx, const EndLoopInsn &i, size_t iinsn)
{
    ctx.loop_depth -= 1;
    if(ctx.loop_depth != 0)
    {
        ctx.indentation -= 4;
        std::fprintf(ctx.file, "%*s}\n", ctx.indentation, " ");
    }
}

static void Lower_Metal(LowerCtx &ctx, const LoadInsn &i, size_t iinsn)
{
    std::fprintf(ctx.file, "%*sfloat v%zu = i%zu[v%zu];\n",
                 ctx.indentation, " ", iinsn, i.input, i.idx);
}

static void Lower_Metal(LowerCtx &ctx, const StoreInsn &i, size_t iinsn)
{
    std::fprintf(ctx.file, "%*soutput[v%zu] = v%zu;\n", ctx.indentation, " ", i.offset, i.value);
}

static void Lower_Metal(LowerCtx &ctx, const LoadImmediateInsn &i, size_t iinsn)
{
    std::fprintf(ctx.file, "%*sfloat v%zu = %f;\n", ctx.indentation, " ", iinsn, i.value);
}

static void Lower_Metal(LowerCtx &ctx, const UnaryInsn &i, size_t iinsn)
{
    auto op_str = i.type == UnaryOpType::EXP ? "exp"
        : i.type == UnaryOpType::LOG ? "log"
        : i.type == UnaryOpType::SIN ? "sin"
        : i.type == UnaryOpType::SQRT ? "sqrtf"
        : "INVALID";
    std::fprintf(ctx.file, "%*sfloat v%zu = %s(v%zu);\n",
                 ctx.indentation, " ", iinsn, op_str, i.x);
}

static void Lower_Metal(LowerCtx &ctx, const BinaryInsn &i, size_t iinsn)
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

        std::fprintf(ctx.file, "%*sfloat v%zu = (float)(v%zu %s v%zu);\n",
                     ctx.indentation, " ", iinsn, i.x, op_str, i.y);
    }
    else if(i.type == BinaryOpType::MAX)
    {
        std::fprintf(ctx.file, "%*sfloat v%zu = v%zu > v%zu ? v%zu : v%zu;\n",
                     ctx.indentation, " ", iinsn, i.x, i.y, i.x, i.y);
    }
    else
    {
        std::fprintf(ctx.file, "%*sfloat v%zu = pow(v%zu, v%zu);\n",
                     ctx.indentation, " ", iinsn, i.x, i.y);
    }
}

static void Lower_Metal(LowerCtx &ctx, const AccumulateInsn &i, size_t iinsn)
{
    if(i.type == ReduceOpType::MAX)
        std::fprintf(ctx.file,
                     "%*sv%zu = v%zu > v%zu ? v%zu : v%zu;\n",
                     ctx.indentation, " ", i.accumulator, i.accumulator, i.x, i.accumulator, i.x);
    else
        std::fprintf(ctx.file, "%*sv%zu += v%zu;\n", ctx.indentation, " ", i.accumulator, i.x);
}

static void Lower_Metal(LowerCtx &ctx, const FunctionBuilder &fn, size_t ifn)
{
    std::fprintf(ctx.file, "kernel void %s_%zu(\n", ctx.prefix, ifn);
    for(size_t i = 0; i < fn.inputs.size(); i++)
        std::fprintf(ctx.file, "    device const float *i%zu,\n", i);
    std::fprintf(ctx.file, "    device float *output,\n");
    std::fprintf(ctx.file, "    uint outer_idx [[thread_position_in_grid]])\n{\n");
    ctx.indentation = 4;
    for(size_t i = 0; i < fn.insns.size(); i++)
    {
        std::visit([&](auto &&insn) { Lower_Metal(ctx, insn, i); }, fn.insns[i]);
    }
    std::fprintf(ctx.file, "}\n\n");
}

using GraphEvalFn = BackendMetal::GraphEvalFn;

static std::pair<GraphEvalFn, void *> CompileAndLoad(const std::filesystem::path &source_path)
{
    std::filesystem::path ir_path = source_path;
    ir_path.replace_extension(".ir");

    std::filesystem::path lib_path = source_path;
    lib_path.replace_extension(".metallib");

    std::string ir_command =
        "xcrun -sdk macosx metal -c " +
        source_path.string() +
        " -o " +
        ir_path.string();
    std::system(ir_command.c_str());

    // std::printf("IR with: %s\n", ir_command.c_str());

    std::string lib_command =
        "xcrun -sdk macosx metallib " +
        ir_path.string() +
        " -o " +
        lib_path.string();
    std::system(lib_command.c_str());

    // std::printf("Metallib with: %s\n", lib_command.c_str());
    auto devices = MTL::CopyAllDevices();
    MTL::Device* mtl_device_ = static_cast<MTL::Device*>(devices->object(0)) ?: MTL::CreateSystemDefaultDevice();

    auto library = NS::String::string(lib_path.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nullptr;
    MTL::Library* library_ = mtl_device_->newLibrary(library, &error);
    if (error != nullptr)
      throw std::runtime_error("failed to create metal library");

#if 0
    void *handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(!handle)
        throw std::runtime_error(dlerror());
    dlerror(); // Clear error conditions
    auto main_fn = reinterpret_cast<GraphEvalFn>(dlsym(handle, "gigagrad_main"));
    if(!main_fn)
    {
        char *err = dlerror();
        if(!err)
            throw std::runtime_error("Symbol gigagrad_main is NULL, which is unexpected");
        else
            throw std::runtime_error(err);
    }
    return { main_fn, handle };
#endif
}

static std::pair<GraphEvalFn, void *> Lower_Metal(const char *prefix, const Program &program)
{
    auto file_name = std::filesystem::temp_directory_path() / prefix;
    file_name += ".metal";
    std::printf("FILE: %s\n", file_name.c_str());
    FILE *file = std::fopen(file_name.c_str(), "w+");
    if(!file)
        throw std::system_error(errno, std::generic_category());

    LowerCtx ctx = { prefix, file, 0, 0 };

    for(size_t ifn = 0; ifn < program.functions.size(); ifn++)
        ::Lower_Metal(ctx, program.functions[ifn], ifn);

    std::fclose(file);
    return CompileAndLoad(file_name);
}

BackendMetal::~BackendMetal()
{
    dlclose(this->handle);
    for(ssize_t ibuff = 0; ibuff < std::ssize(this->program.buffers); ibuff++)
    {
        auto &desc = this->program.buffers[ibuff];
        if(!std::holds_alternative<GraphNodeHandle>(desc.id))
        {
            delete [] reinterpret_cast<float *>(this->buffers[ibuff]);
        }
    }
}

void BackendMetal::LowerProgram(Program &&program)
{
    this->program = std::move(program);
    auto [eval_fn, handle] = Lower_Metal("gg_metal", this->program);
    this->eval_fn = eval_fn;
    this->handle = handle;
}

void *BackendMetal::InitBuffers()
{
    this->buffers.reserve(this->program.buffers.size());
    for(ssize_t ibuff = 0; ibuff < std::ssize(this->program.buffers); ibuff++)
    {
        auto &desc = this->program.buffers[ibuff];
        if(std::holds_alternative<GraphNodeHandle>(desc.id))
        {
            GraphNodeHandle tensor = std::get<GraphNodeHandle>(desc.id);
            this->buffers.push_back(reinterpret_cast<void *>(tensor.data()));
        }
        else
        {
            float *intermediate_buf = new float[desc.size_elts];
            this->buffers.push_back(reinterpret_cast<void *>(intermediate_buf));
        }
    }
    return this->buffers[this->program.functions.back().output_buffer];
}

void *BackendMetal::GetBuffer(size_t idx)
{
    return this->buffers.at(idx);
}

void BackendMetal::Execute()
{
    for(ssize_t ibuff = 0; ibuff < std::ssize(this->program.buffers); ibuff++)
    {
        auto &desc = this->program.buffers[ibuff];
        if(std::holds_alternative<GraphNodeHandle>(desc.id))
        {
            GraphNodeHandle tensor = std::get<GraphNodeHandle>(desc.id);
            this->buffers[ibuff] = (reinterpret_cast<void *>(tensor.data()));
        }
    }
    eval_fn(this->buffers.data());
}
