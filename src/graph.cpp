#include "graph.h"

#include <algorithm>
#include <functional>
#include <numeric>
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

GraphNode &GraphNode::sum(dim_t dim, bool keepdim)
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

GraphNode &GraphNode::max(dim_t dim, bool keepdim)
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

GraphNode &GraphNode::reshape(dim_t length)
{
    return this->reshape(Shape{length});
}

GraphNode &GraphNode::permute(Dims dims)
{
    Graph &graph = GetGraph(*this);
    return graph.AddNode(PermuteOp{graph, *this, std::move(dims)});
}

GraphNode &GraphNode::transpose()
{
    Shape shape = this->shape();
    Dims dims(shape.size());
    std::iota(std::rbegin(dims), std::rend(dims), 0);
    return this->permute(std::move(dims));
}

// axb * bxc -> axc
// ABx1 * 
GraphNode &GraphNode::matmul(GraphNode &x)
{
    
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

GraphNode &operator-(GraphNode &x)
{
    return 0 - x;
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

GraphNode &operator/(GraphNode &x, GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::DIV, x, y});
}

GraphNode &operator/(float x, GraphNode &y)
{
    Graph &graph = GetGraph(y);
    GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode / y;
}

GraphNode &operator/(GraphNode &x, float y)
{
    Graph &graph = GetGraph(x);
    GraphNode &ynode = graph.AddNode(Immediate{graph, y});
    return x / ynode;
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

GraphNode &operator%(GraphNode &x, GraphNode &y)
{
    return x.matmul(y);
}

GraphNode &max(GraphNode &x, float y)
{
    return max(y, x);
}

GraphNode &sum(GraphNode &x, bool keepdim)
{
    return x.sum(keepdim);
}

GraphNode &sum(GraphNode &x, dim_t axis, bool keepdim)
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

GraphNode &max(GraphNode &x, dim_t axis, bool keepdim)
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

GraphNode &reshape(GraphNode &x, dim_t length)
{
    return x.reshape(length);
}

GraphNode &permute(GraphNode &x, Dims permutation)
{
    return x.permute(std::move(permutation));
}

GraphNode &transpose(GraphNode &x)
{
    return x.transpose();
}

GraphNode &matmul(GraphNode &x, GraphNode &y)
{
    return x.matmul(y);
}

GraphNode &Graph::AddTensor(Shape shape)
{
    AddNode(Tensor{*this, std::move(shape)});
    return this->nodes.back();
}

GraphNode &Graph::AddTensor(dim_t dim)
{
    return this->AddTensor(Shape{dim});
}

GraphNode &Graph::AddNode(GraphNode node)
{
    this->nodes.emplace_back(node);
    return this->nodes.back();
}

Shape GetBroadcastedShape(Shape x, Shape y)
{
    // Ensure x.size() >= y.size()
    if(y.size() > x.size())
        std::swap(x, y);

    for(auto i = 0; i < y.size(); i++)
    {
        // Store the proper dimension in dim_x
        auto &dim_x = x[x.size() - i - 1];
        const auto &dim_y = y[y.size() - i - 1];
        if(dim_x == 1 && dim_y != 1)
            dim_x = dim_y;
        else if(dim_x != 1 && dim_y != 1)
            continue;
        else if(dim_x == dim_y)
            continue;
        else
            throw std::domain_error("Cannot broadcast incompatible shapes");
    }
    return x;
}

Shape VerifyWithShape(const GraphNode &node);

Shape VerifyWithShape(const Tensor &t)
{
    return t.shape;
}

Shape VerifyWithShape(const Immediate &i)
{
    return {};
}

Shape VerifyWithShape(const UnaryOp &u)
{
    return VerifyWithShape(u.x);
}

Shape VerifyWithShape(const BinaryOp &u)
{
    Shape shapex = VerifyWithShape(u.x);
    Shape shapey = VerifyWithShape(u.y);
    return GetBroadcastedShape(std::move(shapex), std::move(shapey));
}

Shape VerifyWithShape(const FusedOp &f)
{
    Shape shapex = VerifyWithShape(f.x);
    Shape shapey = VerifyWithShape(f.y);
    Shape shapez = VerifyWithShape(f.z);

    // Broadcast x, y together first because multiplication gets precedence
    Shape shapexy = GetBroadcastedShape(std::move(shapex), std::move(shapey));
    Shape shapexyz = GetBroadcastedShape(std::move(shapexy), shapez);
    return shapexyz;
}

Shape VerifyWithShape(const ReduceOp &r)
{
    Shape shape = VerifyWithShape(r.x);
    if(r.dims.empty())
    {
        if(r.keepdim)
            return Shape(shape.size(), 1); // Shape of all 1's
        return {};
    }

    if(r.dims.size() > shape.size())
        throw std::domain_error("Specified more dims to reduce on than there are dimensions in tensor");
    
    for(auto dim : r.dims)
    {
        if(dim < 0 || dim >= r.dims.size())
            throw std::domain_error("Specified dimension is out of range on reduction operation");
        shape[dim] = 1;
    }
    if(r.keepdim)
        std::remove_if(shape.begin(), shape.end(), [](auto n) { return n == 1; });
    return shape;
}

Shape VerifyWithShape(const ReshapeOp &r)
{
    Shape input_shape = VerifyWithShape(r.x);
    Shape new_shape = r.shape;
    auto num_elements = std::accumulate(input_shape.begin(), input_shape.end(), dim_t{1}, std::multiplies{});
    auto num_implicit_dims = std::count(new_shape.begin(), new_shape.end(), -1);
    if(num_implicit_dims == 0)
    {
        auto new_num_elements = std::accumulate(new_shape.begin(), new_shape.end(), dim_t{1}, std::multiplies{});
        if(new_num_elements != num_elements)
            throw std::domain_error("Reshape number of elements doesn't match that of input tensor");
        return new_shape;
    }

    if(num_implicit_dims > 1)
        throw std::domain_error("Reshape can have at most one implicit dimension");

    auto num_elems_not_including_implicit_dim = std::accumulate(
        new_shape.begin(),
        new_shape.end(),
        dim_t{1},
        [](auto x, auto y)
        {
            if(y == -1)
                return x;
            return x * y;
        });
    auto remaining_dim = num_elements / num_elems_not_including_implicit_dim;
    for(auto &x : new_shape)
        if(x == -1)
            x = remaining_dim;
    return new_shape;
}

Shape VerifyWithShape(const PermuteOp &p)
{
    Shape shape = VerifyWithShape(p.x);
    if(p.dims.size() != shape.size())
        throw std::domain_error("Permute not given proper number of dimensions");
    std::vector<bool> uniqueness(shape.size(), false);
    Shape result(shape.size());
    for(size_t i = 0; i < shape.size(); i++)
    {
        // If dim is negative, we need to fix it to be between 0 and shape.size()
        auto dim = p.dims[i];
        auto fixed_dim = ((dim % shape.size()) + shape.size()) % shape.size();
        if(fixed_dim >= shape.size())
            throw std::domain_error("Dim index out of range");
        if(uniqueness[fixed_dim])
            throw std::domain_error("Found repeated dim in permute");
        uniqueness[fixed_dim] = true;
        result[fixed_dim] = shape[i];
    }
    return result;
}

Shape VerifyWithShape(const GraphNode &node)
{
    return std::visit([](auto &&arg) { return VerifyWithShape(arg); }, node);
}

Shape GraphNode::shape() const
{
    return std::visit([](auto &&arg) { return VerifyWithShape(arg); }, *this);
}

void GraphNode::Verify() const
{
    std::visit([](auto &&arg) { VerifyWithShape(arg); }, *this);
}

}
