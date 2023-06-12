#pragma once

#include <deque>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

namespace Gigagrad
{

struct Tensor;
struct Immediate;
struct UnaryOp;
struct BinaryOp;
struct FusedOp;
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

enum class FusedOpType
{
    FMA,
};

enum class ReduceOpType
{
    SUM,
    MAX,
};

struct Tensor
{
    Graph &graph;
    Shape shape;
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
    GraphNode &x;
};

struct BinaryOp
{
    Graph &graph;
    BinaryOpType type;
    const GraphNode &x;
    const GraphNode &y;
};

struct FusedOp
{
    Graph &graph;
    FusedOpType type;
    const GraphNode &x;
    const GraphNode &y;
    const GraphNode &z;
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

struct GraphNode : std::variant<Tensor, Immediate, UnaryOp, BinaryOp, FusedOp, ReduceOp, ReshapeOp, PermuteOp>
{
    using variant::variant;
    GraphNode &sum(bool keepdim = false);
    GraphNode &sum(dim_t axis, bool keepdim = false);
    GraphNode &sum(Dims dims, bool keepdim = false);
    GraphNode &max(bool keepdim = false);
    GraphNode &max(dim_t axis, bool keepdim = false);
    GraphNode &max(Dims dims, bool keepdim = false);

    GraphNode &reshape(Shape shape);
    GraphNode &reshape(dim_t length);
    GraphNode &permute(Dims dims);
    GraphNode &transpose();

    GraphNode &matmul(GraphNode &y);

    Shape shape() const; // Empty shape means scalar

    void Verify() const;
};

GraphNode &exp(GraphNode &x);
GraphNode &log(GraphNode &x);
GraphNode &sin(GraphNode &x);
GraphNode &operator-(GraphNode &x);

GraphNode &operator+(GraphNode &x, GraphNode &y);
GraphNode &operator+(float x, GraphNode &y);
GraphNode &operator+(GraphNode &x, float y);
GraphNode &operator-(GraphNode &x, GraphNode &y);
GraphNode &operator-(float x, GraphNode &y);
GraphNode &operator-(GraphNode &x, float y);
GraphNode &operator*(GraphNode &x, GraphNode &y);
GraphNode &operator*(float x, GraphNode &y);
GraphNode &operator*(GraphNode &x, float y);
GraphNode &operator/(GraphNode &x, GraphNode &y);
GraphNode &operator/(float x, GraphNode &y);
GraphNode &operator/(GraphNode &x, float y);
GraphNode &operator^(GraphNode &x, float y);
GraphNode &operator==(GraphNode &x, GraphNode &y);
GraphNode &operator==(GraphNode &x, float y);
GraphNode &operator==(float x, GraphNode &y);
GraphNode &max(GraphNode &x, GraphNode &y);
GraphNode &max(float x, GraphNode &y);
GraphNode &max(GraphNode &x, float y);

GraphNode &sum(GraphNode &x, bool keepdim = false);
GraphNode &sum(GraphNode &x, dim_t axis, bool keepdim = false);
GraphNode &sum(GraphNode &x, Dims dims, bool keepdim = false);
GraphNode &max(GraphNode &x, bool keepdim = false);
GraphNode &max(GraphNode &x, dim_t axis, bool keepdim = false);
GraphNode &max(GraphNode &x, Dims dims, bool keepdim = false);

GraphNode &reshape(GraphNode &x, Shape shape);
GraphNode &reshape(GraphNode &x, dim_t length);
GraphNode &permute(GraphNode &x, Dims dims);

GraphNode &operator%(GraphNode &x, GraphNode &y);
GraphNode &matmul(GraphNode &x, GraphNode &y);

struct Graph
{
    GraphNode &AddTensor(Shape shape);
    GraphNode &AddTensor(dim_t dim);
    GraphNode &AddNode(GraphNode node);
    std::deque<GraphNode> nodes;
};

}
