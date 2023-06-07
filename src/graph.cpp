#include "graph.h"

#include <stdexcept>

namespace Gigagrad
{

Graph &GetGraph(GraphNode &x)
{
    return std::visit([](auto &&a) -> Graph & { return a.graph; }, x);
}

GraphNode &WrapInUnary(GraphNode &x, UnaryOpType type)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(UnaryOp{graph, type, x});
}

GraphNode &WrapInReduction(GraphNode &x, ReduceOpType type, Dims dims, bool keepdim)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(ReduceOp{graph, type, x, std::move(dims), keepdim});
}

GraphNode &GraphNode::sum(bool keepdim)
{
    return this->sum(Dims{}, keepdim);
}

GraphNode &GraphNode::sum(size_t dim, bool keepdim)
{
    return this->sum(Dims{dim}, keepdim);
}

GraphNode &GraphNode::sum(Dims dims, bool keepdim)
{
    return WrapInReduction(*this, ReduceOpType::SUM, std::move(dims), keepdim);
}

GraphNode &GraphNode::max(bool keepdim)
{
    return this->max(Dims{}, keepdim);
}

GraphNode &GraphNode::max(size_t dim, bool keepdim)
{
    return this->max(Dims{dim}, keepdim);
}

GraphNode &GraphNode::max(Dims dims, bool keepdim)
{
    return WrapInReduction(*this, ReduceOpType::MAX, std::move(dims), keepdim);
}

GraphNode &GraphNode::reshape(Shape shape)
{
    Graph &graph = GetGraph(*this);
    return graph.AddNode(ReshapeOp{graph, *this, std::move(shape)});
}

GraphNode &GraphNode::reshape(size_t length)
{
    return this->reshape(Shape{length});
}

GraphNode &exp(GraphNode &x)
{
    return WrapInUnary(x, UnaryOpType::EXP);
}

GraphNode &log(GraphNode &x)
{
    return WrapInUnary(x, UnaryOpType::LOG);
}

GraphNode &sin(GraphNode &x)
{
    return WrapInUnary(x, UnaryOpType::SIN);
}

GraphNode &operator+(GraphNode &x, GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::ADD, x, y});
}

GraphNode &operator+(float x, GraphNode &y)
{
    Graph &graph = GetGraph(y);
    GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode + y;
}

GraphNode &operator+(GraphNode &x, float y)
{
    return y + x;
}

GraphNode &operator-(GraphNode &x, GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::SUB, x, y});
}

GraphNode &operator-(float x, GraphNode &y)
{
    return (-x) + y;
}

GraphNode &operator-(GraphNode &x, float y)
{
    return x + (-y);
}

GraphNode &operator*(GraphNode &x, GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::MUL, x, y});
}

GraphNode &operator*(float x, GraphNode &y)
{
    Graph &graph = GetGraph(y);
    GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode * y;
}

GraphNode &operator*(GraphNode &x, float y)
{
    return y * x;
}

GraphNode &operator/(GraphNode &x, float y)
{
    Graph &graph = GetGraph(x);
    GraphNode &ynode = graph.AddNode(Immediate{graph, y});
    return graph.AddNode(BinaryOp{graph, BinaryOpType::DIV, x, ynode});
}

GraphNode &operator^(GraphNode &x, float y)
{
    Graph &graph = GetGraph(x);
    GraphNode &ynode = graph.AddNode(Immediate{graph, y});
    return graph.AddNode(BinaryOp{graph, BinaryOpType::POW, x, ynode});
}

GraphNode &operator==(GraphNode &x, GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::CMP, x, y});
}

GraphNode &operator==(float x, GraphNode &y)
{
    Graph &graph = GetGraph(y);
    GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode == y;
}

GraphNode &operator==(GraphNode &x, float y)
{
    return y == x;
}

GraphNode &max(GraphNode &x, GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::MAX, x, y});
}

GraphNode &max(float x, GraphNode &y)
{
    Graph &graph = GetGraph(y);
    GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return max(xnode, y);
}

GraphNode &max(GraphNode &x, float y)
{
    return max(y, x);
}

GraphNode &sum(GraphNode &x, bool keepdim)
{
    return x.sum(keepdim);
}

GraphNode &sum(GraphNode &x, size_t axis, bool keepdim)
{
    return x.sum(axis, keepdim);
}

GraphNode &sum(GraphNode &x, Dims dims, bool keepdim)
{
    return x.sum(std::move(dims), keepdim);
}

GraphNode &max(GraphNode &x, bool keepdim)
{
    return x.max(keepdim);
}

GraphNode &max(GraphNode &x, size_t axis, bool keepdim)
{
    return x.max(axis, keepdim);
}

GraphNode &max(GraphNode &x, Dims dims, bool keepdim)
{
    return x.max(std::move(dims), keepdim);
}

GraphNode &reshape(GraphNode &x, Shape shape)
{
    return x.reshape(std::move(shape));
}

GraphNode &reshape(GraphNode &x, size_t length)
{
    return x.reshape(length);
}

GraphNode &Graph::AddTensor(Shape shape)
{
    AddNode(Tensor{*this, std::move(shape)});
    return this->nodes.back();
}

GraphNode &Graph::AddNode(GraphNode node)
{
    this->nodes.emplace_back(node);
    return this->nodes.back();
}

Shape ComputeShape(const Tensor &t)
{
    return t.shape;
}

Shape ComputeShape(const Immediate &i)
{
    return {};
}

Shape ComputeShape(const UnaryOp &u)
{
    return u.x.shape();
}

Shape ComputeShape(const BinaryOp &b)
{
    Shape xshape = b.x.shape();
    if(!xshape.empty())
        return xshape;
    return b.y.shape();
}

Shape ComputeShape(const FusedOp &f)
{
    Shape xshape = f.x.shape();
    if(!xshape.empty())
        return xshape;
    Shape yshape = f.y.shape();
    if(!yshape.empty())
        return yshape;
    return f.z.shape();
}

Shape ComputeShape(const ReduceOp &r)
{
    if(r.dims.size() == 0)
        return {};
    
    Shape shape = r.x.shape();
    return shape;
}

Shape ComputeShape(const ReshapeOp &r)
{
    return r.shape;
}

Shape GraphNode::shape() const
{
    return std::visit([](auto &&arg) { return ComputeShape(arg); }, *this);
}

Shape VerifyWithShape(const GraphNode &node);

Shape VerifyWithShape(const Tensor &t)
{
    return ComputeShape(t);
}

Shape VerifyWithShape(const Immediate &i)
{
    return ComputeShape(i);
}

Shape VerifyWithShape(const UnaryOp &u)
{
    return VerifyWithShape(u.x);
}

Shape VerifyWithShape(const BinaryOp &u)
{
    Shape shapex = VerifyWithShape(u.x);
    Shape shapey = VerifyWithShape(u.y);
    if(shapex.empty())
        return shapey;
    if(shapey.empty())
        return shapex;
    if(shapex != shapey)
        throw std::domain_error("Mismatched shape in binary op");
    return shapex;
}

Shape VerifyWithShape(const GraphNode &node)
{
    return std::visit([](auto &&arg) { return VerifyWithShape(arg); }, node);
}

void GraphNode::verify() const
{
    std::visit([](auto &&arg) { VerifyWithShape(arg); }, *this);
}

void GraphNode::codegen(std::ostringstream &s)
{
    this->verify();
}

}
