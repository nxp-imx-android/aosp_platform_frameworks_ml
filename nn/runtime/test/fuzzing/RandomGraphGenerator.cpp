/*
 * Copyright (C) 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fuzzing/RandomGraphGenerator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "TestNeuralNetworksWrapper.h"
#include "fuzzing/OperationManager.h"
#include "fuzzing/RandomGraphGeneratorUtils.h"
#include "fuzzing/RandomVariable.h"

namespace android {
namespace nn {
namespace fuzzing_test {

using test_wrapper::Result;
using test_wrapper::Type;

// Construct a RandomOperand from OperandSignature.
RandomOperand::RandomOperand(const OperandSignature& operand, Type dataType, uint32_t rank)
    : type(operand.type), finalizer(operand.finalizer) {
    NN_FUZZER_LOG << "Operand: " << toString(type);
    if (operand.constructor) operand.constructor(dataType, rank, this);
}

std::vector<uint32_t> RandomOperand::getDimensions() const {
    std::vector<uint32_t> result(dimensions.size(), 0);
    for (uint32_t i = 0; i < dimensions.size(); i++) result[i] = dimensions[i].getValue();
    return result;
}

// Check if an edge between [this, other] is valid. If yes, add the edge.
bool RandomOperand::createEdgeIfValid(const RandomOperand& other) const {
    if (other.type != RandomOperandType::INPUT) return false;
    if (dataType != other.dataType || dimensions.size() != other.dimensions.size() ||
        scale != other.scale || zeroPoint != other.zeroPoint)
        return false;
    return RandomVariableNetwork::get()->setEqualIfCompatible(dimensions, other.dimensions);
}

uint32_t RandomOperand::getNumberOfElements() const {
    uint32_t num = 1;
    for (const auto& d : dimensions) num *= d.getValue();
    return num;
}

size_t RandomOperand::getBufferSize() const {
    return kSizeOfDataType[static_cast<int32_t>(dataType)] * getNumberOfElements();
}

// Construct a RandomOperation from OperationSignature.
RandomOperation::RandomOperation(const OperationSignature& operation)
    : opType(operation.opType), finalizer(operation.finalizer) {
    NN_FUZZER_LOG << "Operation: " << kOperationNames[static_cast<int32_t>(opType)];

    // Determine the data type and rank of the operation and invoke the constructor.
    Type dataType = getRandomChoice(operation.supportedDataTypes);
    uint32_t rank = getRandomChoice(operation.supportedRanks);

    // Initialize operands and operation.
    for (const auto& op : operation.inputs) {
        inputs.emplace_back(new RandomOperand(op, dataType, rank));
    }
    for (const auto& op : operation.outputs) {
        outputs.emplace_back(new RandomOperand(op, dataType, rank));
    }
    if (operation.constructor) operation.constructor(dataType, rank, this);

    // Add constraints on the number of elements.
    if (RandomVariable::defaultValue > 10) {
        for (auto in : inputs) RandomVariableNetwork::get()->addDimensionProd(in->dimensions);
        for (auto out : outputs) RandomVariableNetwork::get()->addDimensionProd(out->dimensions);
    }
    // The output operands should have dimensions larger than 0.
    for (auto out : outputs) {
        for (auto& dim : out->dimensions) dim.setRange(1, kInvalidValue);
    }
}

bool RandomGraph::generate(uint32_t seed, uint32_t numOperations, uint32_t dimensionRange) {
    RandomNumberGenerator::generator.seed(seed);
    // The generator may (with low probability) end up with an invalid graph.
    // If so, regenerate the graph until a valid one is produced.
    while (true) {
        RandomVariableNetwork::get()->initialize(dimensionRange);
        mOperations.clear();
        mOperands.clear();
        if (generateGraph(numOperations) && generateValue()) break;
        std::cout << "[ Retry    ]   The RandomGraphGenerator produces an invalid graph.\n";
    }
    return true;
}

bool RandomGraph::generateGraph(uint32_t numOperations) {
    NN_FUZZER_LOG << "Generate Graph";
    // Randomly generate a vector of operations, this is a valid topological sort.
    for (uint32_t i = 0; i < numOperations; i++) {
        mOperations.emplace_back(OperationManager::get()->getRandomOperation());
    }

    // Randomly add edges from the output of one operation to the input of another operation
    // with larger positional index.
    for (uint32_t i = 0; i < numOperations; i++) {
        for (auto& output : mOperations[i].outputs) {
            for (uint32_t j = i + 1; j < numOperations; j++) {
                for (auto& input : mOperations[j].inputs) {
                    // For each [output, input] pair, add an edge with probability prob.
                    // TODO: Find a better formula by first defining what "better" is.
                    float prob = 0.1f;
                    if (getBernoulli(prob)) {
                        if (output->createEdgeIfValid(*input)) {
                            NN_FUZZER_LOG << "Add edge: operation " << i << " -> " << j;
                            input = output;
                            output->type = RandomOperandType::INTERNAL;
                        }
                    }
                }
            }
        }
    }
    return true;
}

static bool asConstant(const std::shared_ptr<RandomOperand>& operand, float prob = 0.5f) {
    if (operand->type == RandomOperandType::CONST) return true;
    if (operand->type != RandomOperandType::INPUT) return false;
    // Force all scalars to be CONST.
    if (kScalarDataType[static_cast<int32_t>(operand->dataType)]) return true;
    if (getBernoulli(prob)) return true;
    return false;
}

// Freeze the dimensions to a random but valid combination.
// Generate random buffer values for model inputs and constants.
bool RandomGraph::generateValue() {
    NN_FUZZER_LOG << "Generate Value";
    if (!RandomVariableNetwork::get()->freeze()) return false;

    // Fill all unique operands into mOperands.
    std::set<std::shared_ptr<RandomOperand>> seen;
    auto fillOperands = [&seen, this](const std::vector<std::shared_ptr<RandomOperand>>& ops) {
        for (const auto& op : ops) {
            if (seen.find(op) == seen.end()) {
                seen.insert(op);
                mOperands.push_back(op);
            }
        }
    };
    for (const auto& operation : mOperations) {
        fillOperands(operation.inputs);
        fillOperands(operation.outputs);
    }

    // Count the number of INPUTs.
    uint32_t numInputs = 0;
    for (auto& operand : mOperands) {
        if (operand->type == RandomOperandType::INPUT) numInputs++;
    }

    for (auto& operand : mOperands) {
        // Turn INPUT into CONST with probability prob. Need to keep at least one INPUT.
        float prob = 0.5f;
        if (asConstant(operand, prob) && numInputs > 1) {
            if (operand->type == RandomOperandType::INPUT) numInputs--;
            operand->type = RandomOperandType::CONST;
        }
        if (operand->type != RandomOperandType::INTERNAL) {
            if (operand->buffer.empty()) operand->resizeBuffer<uint8_t>(operand->getBufferSize());
            // If operand is set by randomBuffer, copy the frozen values into buffer.
            if (!operand->randomBuffer.empty()) {
                int32_t* data = reinterpret_cast<int32_t*>(operand->buffer.data());
                for (uint32_t i = 0; i < operand->randomBuffer.size(); i++) {
                    data[i] = operand->randomBuffer[i].getValue();
                }
            }
            if (operand->finalizer) operand->finalizer(operand.get());
        }
    }

    for (auto& operation : mOperations) {
        if (operation.finalizer) operation.finalizer(&operation);
    }
    return true;
}

void RandomGraph::createModel(test_wrapper::Model* model) {
    NN_FUZZER_LOG << "Create Model";

    // Set model operands.
    std::vector<uint32_t> modelInputs;
    std::vector<uint32_t> modelOutputs;
    for (auto& operand : mOperands) {
        // TODO: Model operands are always fully-specified at model construction time.
        test_wrapper::OperandType type(operand->dataType, operand->getDimensions(), operand->scale,
                                       operand->zeroPoint);
        operand->opIndex = model->addOperand(&type);

        // For INPUT/OUTPUT, prepare vectors for identifyInputsAndOutputs(...).
        // For CONST, set operand buffer.
        if (operand->type == RandomOperandType::INPUT) {
            operand->ioIndex = modelInputs.size();
            modelInputs.push_back(operand->opIndex);
        } else if (operand->type == RandomOperandType::OUTPUT) {
            operand->ioIndex = modelOutputs.size();
            modelOutputs.push_back(operand->opIndex);
        } else if (operand->type == RandomOperandType::CONST) {
            model->setOperandValue(operand->opIndex, operand->buffer.data(),
                                   operand->getBufferSize());
        }
    }

    // Set model operations.
    for (auto& operation : mOperations) {
        NN_FUZZER_LOG << "Operation: " << kOperationNames[static_cast<int32_t>(operation.opType)];
        std::vector<uint32_t> inputIndices, outputIndices;
        for (auto& op : operation.inputs) {
            NN_FUZZER_LOG << toString(*op);
            inputIndices.push_back(op->opIndex);
        }
        for (auto& op : operation.outputs) {
            NN_FUZZER_LOG << toString(*op);
            outputIndices.push_back(op->opIndex);
        }
        model->addOperation(operation.opType, inputIndices, outputIndices);
    }

    // Set model inputs and outputs.
    model->identifyInputsAndOutputs(modelInputs, modelOutputs);
}

void RandomGraph::createRequest(test_wrapper::Execution* execution) {
    NN_FUZZER_LOG << "Create Request";
    for (auto& operand : mOperands) {
        if (operand->type == RandomOperandType::INPUT) {
            EXPECT_EQ(execution->setInput(operand->ioIndex, operand->buffer.data(),
                                          operand->getBufferSize(), nullptr),
                      Result::NO_ERROR);
        } else if (operand->type == RandomOperandType::OUTPUT) {
            EXPECT_EQ(execution->setOutput(operand->ioIndex, operand->buffer.data(),
                                           operand->getBufferSize(), nullptr),
                      Result::NO_ERROR);
        }
    }
}

void RandomGraph::dumpSpecFile(std::string filename, std::string testname = "") {
    NN_FUZZER_LOG << "Dump Spec File to " << filename;
    SpecWriter writer(filename, testname);
    ASSERT_TRUE(writer.isOpen());
    writer.dump(mOperations, mOperands);
}

}  // namespace fuzzing_test
}  // namespace nn
}  // namespace android