#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

#include "backend.h"

namespace gigagrad
{

struct Tensor;
struct Immediate;
struct UnaryOp;
struct BinaryOp;
struct ReduceOp;
struct ReshapeOp;

struct Graph;
struct GraphNode;
using dim_t = ssize_t;
using Shape = std::vector<dim_t>;
using Dims = std::vector<dim_t>;

enum class UnaryOpType
{
    NOP,
    EXP,
    LOG,
    CAST,
    SIN,
};

enum class BinaryOpType
{
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    CMP,
    MAX,
};

enum class ReduceOpType
{
    SUM,
    MAX,
};

struct Tensor
{
    using InitFn = std::function<void(float *)>;
    using LoadDataFn = std::function<void(float *, size_t)>;
    Graph &graph;
    Shape shape;
    mutable std::optional<InitFn> init;
    mutable std::optional<LoadDataFn> load;
    mutable float *data = nullptr;
};

struct Immediate
{
    Graph &graph;
    float value;
};

struct UnaryOp
{
    Graph &graph;
    UnaryOpType type;
    const GraphNode &x;
};

struct BinaryOp
{
    Graph &graph;
    BinaryOpType type;
    const GraphNode &x;
    const GraphNode &y;
};

struct ReduceOp
{
    Graph &graph;
    ReduceOpType type;
    const GraphNode &x;
    Dims dims;
    bool keepdim;
};

struct ReshapeOp
{
    Graph &graph;
    const GraphNode &x;
    Shape shape;
};

struct PermuteOp
{
    Graph &graph;
    const GraphNode &x;
    Dims dims;
};

struct CompiledTensor
{
    float *data;
    Shape shape;
    std::unique_ptr<codegen::Backend> backend;

    void Execute() { backend->Execute(); }
};

struct GraphNode : std::variant<Tensor, Immediate, UnaryOp, BinaryOp, ReduceOp, ReshapeOp, PermuteOp>
{
    using variant::variant;
    
    const GraphNode &sum(bool keepdim = false) const;
    const GraphNode &sum(dim_t axis, bool keepdim = false) const;
    const GraphNode &sum(Dims dims, bool keepdim = false) const;
    const GraphNode &max(bool keepdim = false) const;
    const GraphNode &max(dim_t axis, bool keepdim = false) const;
    const GraphNode &max(Dims dims, bool keepdim = false) const;

    const GraphNode &reshape(Shape shape) const;
    const GraphNode &reshape(dim_t length) const;
    const GraphNode &permute(Dims dims) const;
    const GraphNode &transpose() const;

    const GraphNode &matmul(const GraphNode &y) const;

    // TODO: Cache this. I know it's terrible that we walk the tree every time
    Shape shape() const; // Empty shape means scalar

    void Verify() const;

    float *&data() { return std::get<Tensor>(*this).data; }

    CompiledTensor Compile(std::unique_ptr<codegen::Backend> backend) const;
    template <typename TBackend>
    CompiledTensor Compile() const { return Compile(std::make_unique<TBackend>()); }
};

const GraphNode &exp(const GraphNode &x);
const GraphNode &log(const GraphNode &x);
const GraphNode &sin(const GraphNode &x);
const GraphNode &sigmoid(const GraphNode &x);
const GraphNode &operator-(const GraphNode &x);

const GraphNode &operator+(const GraphNode &x, const GraphNode &y);
const GraphNode &operator+(float x, const GraphNode &y);
const GraphNode &operator+(const GraphNode &x, float y);
const GraphNode &operator-(const GraphNode &x, const GraphNode &y);
const GraphNode &operator-(float x, const GraphNode &y);
const GraphNode &operator-(const GraphNode &x, float y);
const GraphNode &operator*(const GraphNode &x, const GraphNode &y);
const GraphNode &operator*(float x, const GraphNode &y);
const GraphNode &operator*(const GraphNode &x, float y);
const GraphNode &operator/(const GraphNode &x, const GraphNode &y);
const GraphNode &operator/(float x, const GraphNode &y);
const GraphNode &operator/(const GraphNode &x, float y);
const GraphNode &operator^(const GraphNode &x, float y);
const GraphNode &operator==(const GraphNode &x, const GraphNode &y);
const GraphNode &operator==(const GraphNode &x, float y);
const GraphNode &operator==(float x, const GraphNode &y);
const GraphNode &operator<(const GraphNode &x, float y);
const GraphNode &operator<(float x, const GraphNode &y);
const GraphNode &operator<(const GraphNode &x, const GraphNode &y);
const GraphNode &operator<=(const GraphNode &x, float y);
const GraphNode &operator<=(float x, const GraphNode &y);
const GraphNode &operator<=(const GraphNode &x, const GraphNode &y);
const GraphNode &operator>(const GraphNode &x, float y);
const GraphNode &operator>(float x, const GraphNode &y);
const GraphNode &operator>(const GraphNode &x, const GraphNode &y);
const GraphNode &operator>=(const GraphNode &x, float y);
const GraphNode &operator>=(float x, const GraphNode &y);
const GraphNode &operator>=(const GraphNode &x, const GraphNode &y);
const GraphNode &max(const GraphNode &x, const GraphNode &y);
const GraphNode &max(float x, const GraphNode &y);
const GraphNode &max(const GraphNode &x, float y);
const GraphNode &min(const GraphNode &x, const GraphNode &y);
const GraphNode &min(float x, const GraphNode &y);
const GraphNode &min(const GraphNode &x, float y);

const GraphNode &sum(const GraphNode &x, bool keepdim = false);
const GraphNode &sum(const GraphNode &x, dim_t axis, bool keepdim = false);
const GraphNode &sum(const GraphNode &x, Dims dims, bool keepdim = false);
const GraphNode &max(const GraphNode &x, bool keepdim = false);
const GraphNode &max(const GraphNode &x, dim_t axis, bool keepdim = false);
const GraphNode &max(const GraphNode &x, Dims dims, bool keepdim = false);
const GraphNode &min(const GraphNode &x, bool keepdim = false);
const GraphNode &min(const GraphNode &x, dim_t axis, bool keepdim = false);
const GraphNode &min(const GraphNode &x, Dims dims, bool keepdim = false);

const GraphNode &reshape(const GraphNode &x, Shape shape);
const GraphNode &reshape(const GraphNode &x, dim_t length);
const GraphNode &permute(const GraphNode &x, Dims dims);

const GraphNode &operator%(const GraphNode &x, const GraphNode &y);
const GraphNode &matmul(const GraphNode &x, const GraphNode &y);

struct Graph
{
    const GraphNode &AddInput(Shape shape);
    const GraphNode &AddInput(dim_t dim);

    const GraphNode &AddWeight(Shape shape);
    const GraphNode &AddWeight(dim_t dim);

    const GraphNode &AddNode(GraphNode node);
    std::deque<GraphNode> inputs;
    std::deque<GraphNode> weights;
    std::deque<GraphNode> nodes;
};

}
