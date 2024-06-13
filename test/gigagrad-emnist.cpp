#include "src/training.h"
#include "src/backend_scalar_c.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

namespace gg = gigagrad;

// Download dataset from
// https://www.nist.gov/itl/products-and-services/emnist-dataset

enum class DataType : uint8_t
{
    U8 = 0x08,
    I8 = 0x09,
    I16 = 0x0B,
    I32 = 0x0C,
    F32 = 0x0D,
    F64 = 0x0E,
};

size_t SizeOf(DataType dtype)
{
    switch(dtype)
    {
    case DataType::U8:
        return sizeof(uint8_t);
    case DataType::I8:
        return sizeof(int8_t);
    case DataType::I16:
        return sizeof(int16_t);
    case DataType::I32:
        return sizeof(int32_t);
    case DataType::F32:
        return sizeof(float);
    case DataType::F64:
        return sizeof(double);
    default:
        exit(1);
    }
}

struct ParsedDataFile
{
    DataType dtype;
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
};

ParsedDataFile LoadDataFile(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if(!file)
    {
        fprintf(stderr, "File %s not found\n", filename);
        exit(1);
    }
    uint32_t magic = 0;
    if(fseek(file, 0, SEEK_SET))
    {
        fprintf(stderr, "Failed to seek: %d\n", ferror(file));
        exit(1);
    }
    if(fread(&magic, sizeof(uint32_t), 1, file) < 1)
    {
        fprintf(stderr, "Failed to read from file: %d\n", ferror(file));
        exit(1);
    }
    if(static_cast<uint16_t>(magic & 0xFFFF) != 0)
    {
        fprintf(stderr, "Invalid magic bytes at beginning of file %s (0x%x), exiting\n", filename, magic);
        exit(1);
    }

    uint8_t raw_dtype = (magic >> 16) & 0xFF;
    DataType dtype;
    switch(raw_dtype)
    {
    case (uint8_t)DataType::U8:
        dtype = DataType::U8;
        break;
    case (uint8_t)DataType::I8:
        dtype = DataType::I8;
        break;
    case (uint8_t)DataType::I16:
        dtype = DataType::I16;
        break;
    case (uint8_t)DataType::I32:
        dtype = DataType::I32;
        break;
    case (uint8_t)DataType::F32:
        dtype = DataType::F32;
        break;
    case (uint8_t)DataType::F64:
        dtype = DataType::F64;
        break;
    default:
        fprintf(stderr, "Invalid datatype 0x%x, exiting\n", raw_dtype);
        exit(1);
        __builtin_unreachable();
    }


    uint8_t ndim = (magic >> 24) & 0xFF;

    std::vector<size_t> shape(ndim);
    size_t offset = 4;
    size_t total_dataset_size = 1;
    for(size_t i = 0; i < ndim; i++)
    {
        uint32_t dim = 0;
        if(fseek(file, offset, SEEK_SET))
        {
            fprintf(stderr, "Failed to seek to offset %zu\n", offset);
            exit(1);
        }
        if(fread(&dim, sizeof(dim), 1, file) < 1)
        {
            fprintf(stderr, "Failed to read\n");
            exit(1);
        }
        shape[i] = static_cast<size_t>(__builtin_bswap32(dim));
        total_dataset_size *= shape[i];
        offset += sizeof(dim);
    }

    std::vector<uint8_t> data(total_dataset_size * SizeOf(dtype), 0);
    if(fseek(file, offset, SEEK_SET))
    {
        fprintf(stderr, "Failed to seek to offset %zu\n", offset);
        exit(1);
    }
    if(fread(data.data(), SizeOf(dtype), total_dataset_size, file) < total_dataset_size)
    {
        fprintf(stderr, "Failed to read dataset\n");
        exit(1);
    }

    return { dtype, std::move(shape), std::move(data) };
}

template <typename T>
std::vector<float> CastToFloatAndNormalize(const std::vector<uint8_t> &input)
{
    size_t num_elements = input.size() / sizeof(T);
    std::vector<float> result(num_elements);
    const T *input_data = reinterpret_cast<const T *>(input.data());
    for(size_t i = 0; i < num_elements; i++)
    {
        result[i] = static_cast<float>(input_data[i]) / 255.0f;
        result[i] = (result[i] - 0.5f) / 0.5f;
    }
    return result;
}

std::vector<float> CastToFloatAndNormalize(DataType dtype, const std::vector<uint8_t> &input)
{
    switch(dtype)
    {
    case DataType::U8:
        return CastToFloatAndNormalize<uint8_t>(input);
    case DataType::I8:
        return CastToFloatAndNormalize<int8_t>(input);
    case DataType::I16:
        return CastToFloatAndNormalize<int16_t>(input);
    case DataType::I32:
        return CastToFloatAndNormalize<int32_t>(input);
    case DataType::F32:
        return CastToFloatAndNormalize<float>(input);
    case DataType::F64:
        return CastToFloatAndNormalize<double>(input);
    default:
        exit(1);
    }
}

template <typename T>
std::vector<float> ToOneHot(const std::vector<uint8_t> &input)
{
    static_assert(std::is_integral_v<T>, "Labels must be integral");
    size_t num_elements = input.size() / sizeof(T);
    const T *input_data = reinterpret_cast<const T *>(input.data());
    T min_val = std::numeric_limits<T>::max();
    T max_val = 0;
    for(size_t i = 0; i < num_elements; i++)
    {
        if(input_data[i] < 0)
        {
            fprintf(stderr, "Tried to one-hot encode invalid value: %d\n", (int)input_data[i]);
            exit(1);
        }
        max_val = std::max(max_val, input_data[i]);
        min_val = std::min(min_val, input_data[i]);
    }
    T range = (max_val - min_val + 1);
    std::vector<float> result(range * num_elements, 0.0f);
    for(size_t i = 0; i < num_elements; i++)
    {
        T cur_class = input_data[i] - min_val;
        result[i * range + cur_class] = 1.0f;
    }
    return result;
}

std::vector<float> ToOneHot(DataType dtype, const std::vector<uint8_t> &input)
{
    switch(dtype)
    {
    case DataType::U8:
        return ToOneHot<uint8_t>(input);
    case DataType::I8:
        return ToOneHot<int8_t>(input);
    case DataType::I16:
        return ToOneHot<int16_t>(input);
    case DataType::I32:
        return ToOneHot<int32_t>(input);
    default:
        fprintf(stderr, "Can't cast non-integer to one-hot\n");
        exit(1);
    }
}

struct TrainingSet
{
    std::vector<size_t> shape;
    std::vector<float> inputs;
    std::vector<float> labels;
};

struct TestSet
{
    std::vector<size_t> shape;
    std::vector<float> inputs;
    std::vector<uint8_t> labels;
};

std::pair<ParsedDataFile, ParsedDataFile> LoadDataset(const char *directory, const char *dataset)
{
    std::string image_name = std::string(directory) + "/emnist-mnist-" + dataset + "-images-idx3-ubyte";
    std::string label_name = std::string(directory) + "/emnist-mnist-" + dataset + "-labels-idx1-ubyte";
    ParsedDataFile images = LoadDataFile(image_name.c_str());
    ParsedDataFile labels = LoadDataFile(label_name.c_str());
    return { std::move(images), std::move(labels) };
}

TrainingSet LoadTrainingSet(const char *directory)
{
    auto [images, labels] = LoadDataset(directory, "train");
    std::vector<float> images_float = CastToFloatAndNormalize(images.dtype, images.data);
    std::vector<float> labels_onehot = ToOneHot(labels.dtype, labels.data);
    return { std::move(images.shape), std::move(images_float), std::move(labels_onehot) };
}

TestSet LoadTestSet(const char *directory)
{
    auto [images, labels] = LoadDataset(directory, "test");
    std::vector<float> images_float = CastToFloatAndNormalize(images.dtype, images.data);
    return { std::move(images.shape), std::move(images_float), std::move(labels.data) };
}

void InitializeWeights(float *weight, size_t size_elts)
{
    std::default_random_engine gen(0);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(size_t i = 0; i < size_elts; i++)
    {
        do
        {
            weight[i] = dist(gen);
        } while(weight[i] == 0.0f);
    }
}

int main(int argc, const char **argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "Please specify exactly one argument: the directory of the EMNIST dataset\n");
        exit(1);
    }

    TrainingSet train = LoadTrainingSet(argv[1]);

    constexpr size_t NumEpochs = 4;
    constexpr size_t HiddenLayerSize = 128;
    constexpr gg::dim_t BatchSize = 64;

    gg::nn::Module network;
    auto x = network.AddInput({ BatchSize, 28 * 28, 1 });
    auto w1 = network.AddWeight({ HiddenLayerSize, 28 * 28 });
    auto b1 = network.AddWeight({ HiddenLayerSize, 1 });
    auto z1 = (w1 % x) + b1;
    auto a2 = z1.relu();
    auto w2 = network.AddWeight({ 10, HiddenLayerSize });
    auto b2 = network.AddWeight({ 10, 1 });
    auto z2 = (w2 % a2) + b2;
    auto result = z2;

    auto training_example = network.AddInput({ BatchSize, 10, 1 });
    auto loss = CrossEntropyLoss(result, training_example);

    gg::TrainingContext ctx = gg::CompileTrainingGraph<gg::codegen::BackendScalarC>(
        network,
        loss,
        0.001);

    network.graph.ToDotFile("gg_emnist.dot");

    w1.data() = new float[HiddenLayerSize * 28 * 28];
    b1.data() = new float[HiddenLayerSize * 1];
    w2.data() = new float[10 * HiddenLayerSize];
    b2.data() = new float[10 * 1];
    InitializeWeights(w1.data(), HiddenLayerSize * 28 * 28);
    InitializeWeights(b1.data(), HiddenLayerSize * 1);
    InitializeWeights(w2.data(), 10 * HiddenLayerSize);
    InitializeWeights(b2.data(), 10 * 1);

    size_t num_batches = train.shape[0] / BatchSize;
    for(size_t iepoch = 0; iepoch < NumEpochs; iepoch++)
    {
        float epoch_loss = 0.0f;
        for(size_t ibatch = 0; ibatch < num_batches; ibatch++)
        {
            x.data() = &train.inputs[BatchSize * 28 * 28 * ibatch];
            training_example.data() = &train.labels[BatchSize * 10 * ibatch];

            ctx.Execute();

            epoch_loss += *ctx.loss;
            printf("Epoch %zu Batch (%zu / %zu) loss: %.6f\n", iepoch, ibatch, num_batches, *ctx.loss);
        }
        printf("Epoch [%zu/%zu] loss: %.6f\n", iepoch + 1, NumEpochs, epoch_loss / (num_batches * BatchSize));
    }

    TestSet test = LoadTestSet(argv[1]);
    auto eval = result.Compile<gg::codegen::BackendScalarC>();
    size_t num_batches_test = test.shape[0] / BatchSize;
    size_t num_correct = 0;
    for(size_t ibatch = 0; ibatch < num_batches_test; ibatch++)
    {
        x.data() = &test.inputs[BatchSize * 28 * 28 * ibatch];
        eval.Execute();
        for(size_t iexample = 0; iexample < BatchSize; iexample++)
        {
            float *prediction_probabilities = eval.data + iexample * 10;
            uint8_t prediction = std::max_element(prediction_probabilities, prediction_probabilities + 10) - prediction_probabilities;
            uint8_t actual = test.labels[ibatch * BatchSize + iexample];
            num_correct += prediction == actual;
        }
    }
    printf(
        "Evaluation: %zu correct / %zu examples, %.2f%%\n",
        num_correct,
        num_batches_test * BatchSize,
        static_cast<float>(num_correct) / static_cast<float>(num_batches_test * BatchSize) * 100.0f);

    return 0;
}
