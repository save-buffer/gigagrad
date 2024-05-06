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

struct Graph;
struct GraphNode;
using dim_t = ssize_t;
using Shape = std::vector<dim_t>;
using Dims = std::vector<dim_t>;

struct CompiledTensor
{
    float *data;
    Shape shape;
    std::unique_ptr<codegen::Backend> backend;

    void Execute() { backend->Execute(); }
};

struct GraphNodeHandle
{
    Graph *graph;
    size_t node_idx;

    GraphNodeHandle sum(bool keepdim = false) const;
    GraphNodeHandle sum(dim_t axis, bool keepdim = false) const;
    GraphNodeHandle sum(Dims dims, bool keepdim = false) const;
    GraphNodeHandle max(bool keepdim = false) const;
    GraphNodeHandle max(dim_t axis, bool keepdim = false) const;
    GraphNodeHandle max(Dims dims, bool keepdim = false) const;

    GraphNodeHandle reshape(Shape shape) const;
    GraphNodeHandle reshape(dim_t length) const;
    GraphNodeHandle permute(Dims dims) const;
    GraphNodeHandle swapaxes(dim_t axis1, dim_t axis2) const;
    GraphNodeHandle transpose() const;
    GraphNodeHandle as_strided(Shape shape, Shape strides, dim_t offset) const;

    GraphNodeHandle relu() const;
    GraphNodeHandle softmax(dim_t axis = -1) const;

    GraphNodeHandle matmul(GraphNodeHandle y) const;

    const Shape &shape() const; // Empty shape means scalar
    const Shape &strides() const;

    GraphNode &GetNode();
    const GraphNode &GetNode() const;
    GraphNode &operator*() { return this->GetNode(); }
    const GraphNode &operator*() const { return this->GetNode(); }
    GraphNode *operator->() { return &this->GetNode(); }
    const GraphNode *operator->() const { return &this->GetNode(); }
    float *&data();

    CompiledTensor Compile(std::unique_ptr<codegen::Backend> backend) const;
    template <typename TBackend>
    CompiledTensor Compile() const { return Compile(std::make_unique<TBackend>()); }
};

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
    float *data = nullptr;
};

struct Immediate
{
    float value;
};

struct UnaryOp
{
    UnaryOpType type;
    GraphNodeHandle x;
};

struct BinaryOp
{
    BinaryOpType type;
    GraphNodeHandle x;
    GraphNodeHandle y;
};

struct ReduceOp
{
    ReduceOpType type;
    GraphNodeHandle x;
    Dims dims;
    bool keepdim;
};

struct ViewOp
{
    GraphNodeHandle x;
    Shape shape;
    Shape strides;
    dim_t offset;
};

struct GraphNode
{
    enum class Kind
    {
        Tensor,
        Immediate,
        UnaryOp,
        BinaryOp,
        ReduceOp,
        ViewOp,
    };

    union U
    {
        struct { Kind kind; } k;
        struct { Kind kind; Tensor tensor; } t;
        struct { Kind kind; Immediate immediate; } i;
        struct { Kind kind; UnaryOp unary_op; } u;
        struct { Kind kind; BinaryOp binary_op; } b;
        struct { Kind kind; ReduceOp reduce_op; } r;
        struct { Kind kind; ViewOp view_op; } v;

        U(Tensor tensor) : t({ .kind = Kind::Tensor, .tensor = std::move(tensor) }) {}
        U(Immediate immediate) : i({ .kind = Kind::Immediate, .immediate = std::move(immediate) }) {}
        U(UnaryOp unary_op) : u({ .kind = Kind::UnaryOp, .unary_op = std::move(unary_op) }) {}
        U(BinaryOp binary_op) : b({ .kind = Kind::BinaryOp, .binary_op = std::move(binary_op) }) {}
        U(ReduceOp reduce_op) : r({ .kind = Kind::ReduceOp, .reduce_op = std::move(reduce_op) }) {}
        U(ViewOp view_op) : v({ .kind = Kind::ViewOp, .view_op = std::move(view_op) }) {}

        U(const U &that);
        U(U &&that);
        U &operator=(const U &that);
        U &operator=(U &&that);
        ~U();
    };

    template <typename T>
    decltype(auto) Visit(T fn)
    {
        switch(this->u.k.kind)
        {
        case Kind::Tensor:
            return fn(this->u.t.tensor);
        case Kind::Immediate:
            return fn(this->u.i.immediate);
        case Kind::UnaryOp:
            return fn(this->u.u.unary_op);
        case Kind::BinaryOp:
            return fn(this->u.b.binary_op);
        case Kind::ReduceOp:
            return fn(this->u.r.reduce_op);
        case Kind::ViewOp:
            return fn(this->u.v.view_op);
        default:
            throw std::logic_error("Invalid node type! This is a bug");
        }
    }

    Kind Kind() { return this->u.k.kind; }

    U u;
    Shape shape;
    Shape strides;
};

GraphNodeHandle exp(GraphNodeHandle x);
GraphNodeHandle log(GraphNodeHandle x);
GraphNodeHandle sin(GraphNodeHandle x);
GraphNodeHandle cos(GraphNodeHandle x);
GraphNodeHandle sigmoid(GraphNodeHandle x);
GraphNodeHandle operator-(GraphNodeHandle x);

GraphNodeHandle operator+(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator+(float x, GraphNodeHandle y);
GraphNodeHandle operator+(GraphNodeHandle x, float y);
GraphNodeHandle operator-(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator-(float x, GraphNodeHandle y);
GraphNodeHandle operator-(GraphNodeHandle x, float y);
GraphNodeHandle operator*(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator*(float x, GraphNodeHandle y);
GraphNodeHandle operator*(GraphNodeHandle x, float y);
GraphNodeHandle operator/(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator/(float x, GraphNodeHandle y);
GraphNodeHandle operator/(GraphNodeHandle x, float y);
GraphNodeHandle operator^(GraphNodeHandle x, float y);
GraphNodeHandle operator==(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator==(GraphNodeHandle x, float y);
GraphNodeHandle operator==(float x, GraphNodeHandle y);
GraphNodeHandle operator<(GraphNodeHandle x, float y);
GraphNodeHandle operator<(float x, GraphNodeHandle y);
GraphNodeHandle operator<(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator<=(GraphNodeHandle x, float y);
GraphNodeHandle operator<=(float x, GraphNodeHandle y);
GraphNodeHandle operator<=(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator>(GraphNodeHandle x, float y);
GraphNodeHandle operator>(float x, GraphNodeHandle y);
GraphNodeHandle operator>(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle operator>=(GraphNodeHandle x, float y);
GraphNodeHandle operator>=(float x, GraphNodeHandle y);
GraphNodeHandle operator>=(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle max(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle max(float x, GraphNodeHandle y);
GraphNodeHandle max(GraphNodeHandle x, float y);
GraphNodeHandle min(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle min(float x, GraphNodeHandle y);
GraphNodeHandle min(GraphNodeHandle x, float y);
GraphNodeHandle pow(GraphNodeHandle x, float y);
GraphNodeHandle pow(float x, GraphNodeHandle y);
GraphNodeHandle pow(GraphNodeHandle x, GraphNodeHandle y);

GraphNodeHandle relu(GraphNodeHandle x);
GraphNodeHandle softmax(GraphNodeHandle x, dim_t axis = -1);

GraphNodeHandle sum(GraphNodeHandle x, bool keepdim = false);
GraphNodeHandle sum(GraphNodeHandle x, dim_t axis, bool keepdim = false);
GraphNodeHandle sum(GraphNodeHandle x, Dims dims, bool keepdim = false);
GraphNodeHandle max(GraphNodeHandle x, bool keepdim = false);
GraphNodeHandle max(GraphNodeHandle x, dim_t axis, bool keepdim = false);
GraphNodeHandle max(GraphNodeHandle x, Dims dims, bool keepdim = false);
GraphNodeHandle min(GraphNodeHandle x, bool keepdim = false);
GraphNodeHandle min(GraphNodeHandle x, dim_t axis, bool keepdim = false);
GraphNodeHandle min(GraphNodeHandle x, Dims dims, bool keepdim = false);

GraphNodeHandle reshape(GraphNodeHandle x, Shape shape);
GraphNodeHandle reshape(GraphNodeHandle x, dim_t length);
GraphNodeHandle permute(GraphNodeHandle x, Dims dims);

GraphNodeHandle operator%(GraphNodeHandle x, GraphNodeHandle y);
GraphNodeHandle matmul(GraphNodeHandle x, GraphNodeHandle y);

struct Graph
{
    GraphNodeHandle Immediate(float imm);
    GraphNodeHandle AddInput(Shape shape);
    GraphNodeHandle AddInput(dim_t dim);

    GraphNodeHandle AddNode(struct Tensor, Shape shape);
    GraphNodeHandle AddNode(struct Immediate);
    GraphNodeHandle AddNode(struct UnaryOp);
    GraphNodeHandle AddNode(struct BinaryOp);
    GraphNodeHandle AddNode(struct ReduceOp);
    GraphNodeHandle AddNode(struct ViewOp);

    GraphNodeHandle AddNode(GraphNode node);

    std::vector<size_t> inputs;
    std::deque<GraphNode> nodes;
};

namespace nn
{

struct Module
{
    GraphNodeHandle Immediate(float imm);

    GraphNodeHandle AddInput(Shape shape);
    GraphNodeHandle AddInput(dim_t dim);

    GraphNodeHandle AddWeight(Shape shape);
    GraphNodeHandle AddWeight(dim_t dim);
    
    Graph graph;
    // TODO: Think about if this is a good idea..
    // kind of cumbersome doing a double lookup
    std::vector<size_t> weights; // Indices of forward.inputs that are weights
};

}
}
