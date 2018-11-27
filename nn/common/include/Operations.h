/*
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_H
#define ANDROID_ML_NN_COMMON_OPERATIONS_H

#include "operations/Cast.h"
#include "operations/EmbeddingLookup.h"
#include "operations/ExpandDims.h"
#include "operations/HashtableLookup.h"
#include "operations/LSHProjection.h"
#include "operations/LSTM.h"
#include "operations/MaximumMinimum.h"
#include "operations/Multinomial.h"
#include "operations/Pow.h"
#include "operations/QuantizedLSTM.h"
#include "operations/RNN.h"
#include "operations/SVDF.h"
#include "operations/Slice.h"
#include "operations/Tile.h"
#include "operations/TopK_V2.h"

#include <stddef.h>

#include <cstdint>
#include <vector>

namespace android {
namespace nn {

struct Shape;

bool addFloat16(const _Float16* in1, const Shape& shape1, const _Float16* in2, const Shape& shape2,
                int32_t activation, _Float16* out, const Shape& shapeOut);
bool addFloat32(const float* in1, const Shape& shape1, const float* in2, const Shape& shape2,
                int32_t activation, float* out, const Shape& shapeOut);
bool addQuant8(const uint8_t* in1, const Shape& shape1, const uint8_t* in2, const Shape& shape2,
               int32_t activation, uint8_t* out, const Shape& shapeOut);

bool mulFloat16(const _Float16* in1, const Shape& shape1, const _Float16* in2, const Shape& shape2,
                int32_t activation, _Float16* out, const Shape& shapeOut);
bool mulFloat32(const float* in1, const Shape& shape1, const float* in2, const Shape& shape2,
                int32_t activation, float* out, const Shape& shapeOut);
bool mulQuant8(const uint8_t* in1, const Shape& shape1, const uint8_t* in2, const Shape& shape2,
               int32_t activation, uint8_t* out, const Shape& shapeOut);

bool floorFloat32(const float* inputData, float* outputData, const Shape& shape);

bool dequantizeQuant8ToFloat32(const uint8_t* inputData, float* outputData, const Shape& shape);

bool quantizeFloat32ToQuant8(const float* inputData, uint8_t* outputData, const Shape& outputShape);

bool depthwiseConvFloat16(const _Float16* inputData, const Shape& inputShape,
                          const _Float16* filterData, const Shape& filterShape,
                          const _Float16* biasData, const Shape& biasShape, int32_t paddingLeft,
                          int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
                          int32_t strideWidth, int32_t strideHeight, int32_t depthMultiplier,
                          int32_t activation, _Float16* outputData, const Shape& outputShape);
bool depthwiseConvFloat32(const float* inputData, const Shape& inputShape, const float* filterData,
                          const Shape& filterShape, const float* biasData, const Shape& biasShape,
                          int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
                          int32_t paddingBottom, int32_t strideWidth, int32_t strideHeight,
                          int32_t depthMultiplier, int32_t activation, float* outputData,
                          const Shape& outputShape);
bool depthwiseConvQuant8(const uint8_t* inputData, const Shape& inputShape,
                         const uint8_t* filterData, const Shape& filterShape,
                         const int32_t* biasData, const Shape& biasShape, int32_t paddingLeft,
                         int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
                         int32_t strideWidth, int32_t strideHeight, int32_t depthMultiplier,
                         int32_t activation, uint8_t* outputData, const Shape& outputShape);

bool convFloat32(const float* inputData, const Shape& inputShape, const float* filterData,
                 const Shape& filterShape, const float* biasData, const Shape& biasShape,
                 int32_t padding_left, int32_t padding_right, int32_t padding_top,
                 int32_t padding_bottom, int32_t stride_width, int32_t stride_height,
                 int32_t activation, float* outputData, const Shape& outputShape);
bool convQuant8(const uint8_t* inputData, const Shape& inputShape, const uint8_t* filterData,
                const Shape& filterShape, const int32_t* biasData, const Shape& biasShape,
                int32_t padding_left, int32_t padding_right, int32_t padding_top,
                int32_t padding_bottom, int32_t stride_width, int32_t stride_height,
                int32_t activation, uint8_t* outputData, const Shape& outputShape);

bool averagePoolFloat32(const float* inputData, const Shape& inputShape, int32_t padding_left,
                        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                        int32_t stride_width, int32_t stride_height, int32_t filter_width,
                        int32_t filter_height, int32_t activation, float* outputData,
                        const Shape& outputShape);
bool averagePoolQuant8(const uint8_t* inputData, const Shape& inputShape, int32_t padding_left,
                       int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                       int32_t stride_width, int32_t stride_height, int32_t filter_width,
                       int32_t filter_height, int32_t activation, uint8_t* outputData,
                       const Shape& outputShape);
bool l2PoolFloat32(const float* inputData, const Shape& inputShape, int32_t padding_left,
                   int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                   int32_t stride_width, int32_t stride_height, int32_t filter_width,
                   int32_t filter_height, int32_t activation, float* outputData,
                   const Shape& outputShape);
bool maxPoolFloat32(const float* inputData, const Shape& inputShape, int32_t padding_left,
                    int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                    int32_t stride_width, int32_t stride_height, int32_t filter_width,
                    int32_t filter_height, int32_t activation, float* outputData,
                    const Shape& outputShape);
bool maxPoolQuant8(const uint8_t* inputData, const Shape& inputShape, int32_t padding_left,
                   int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                   int32_t stride_width, int32_t stride_height, int32_t filter_width,
                   int32_t filter_height, int32_t activation, uint8_t* outputData,
                   const Shape& outputShape);

template <typename T>
bool reluFloat(const T* inputData, const Shape& inputShape, T* outputData, const Shape& outputShape,
               float reluMin = 0.f, float reluMax = std::numeric_limits<float>::max());
template <typename T>
bool relu1Float(const T* inputData, const Shape& inputShape, T* outputData,
                const Shape& outputShape);
template <typename T>
bool relu6Float(const T* inputData, const Shape& inputShape, T* outputData,
                const Shape& outputShape);

bool tanhFloat16(const _Float16* inputData, const Shape& inputShape, _Float16* outputData,
                 const Shape& outputShape);
bool tanhFloat32(const float* inputData, const Shape& inputShape, float* outputData,
                 const Shape& outputShape);
bool tanhQuant8(const uint8_t* inputData, const Shape& inputShape, uint8_t* outputData,
                const Shape& outputShape);

template <typename T>
bool logisticFloat(const T* inputData, const Shape& inputShape, T* outputData,
                   const Shape& outputShape);

bool softmaxFloat16(const _Float16* inputData, const Shape& inputShape, const float beta,
                    int32_t axis, _Float16* outputData, const Shape& outputShape);
bool softmaxFloat32(const float* inputData, const Shape& inputShape, const float beta, int32_t axis,
                    float* outputData, const Shape& outputShape);

bool reluQuant8(const uint8_t* inputData, const Shape& inputShape, uint8_t* outputData,
                const Shape& outputShape);
bool relu1Quant8(const uint8_t* inputData, const Shape& inputShape, uint8_t* outputData,
                 const Shape& outputShape);
bool relu6Quant8(const uint8_t* inputData, const Shape& inputShape, uint8_t* outputData,
                 const Shape& outputShape);
bool logisticQuant8(const uint8_t* inputData, const Shape& inputShape, uint8_t* outputData,
                    const Shape& outputShape);
bool softmaxQuant8(const uint8_t* inputData, const Shape& inputShape, const float beta,
                   int32_t axis, uint8_t* outputData, const Shape& outputShape);

bool fullyConnectedFloat32(const float* inputData, const Shape& inputShape, const float* weights,
                           const Shape& weightsShape, const float* biasData, const Shape& biasShape,
                           int32_t activation, float* outputData, const Shape& outputShape);
bool fullyConnectedQuant8(const uint8_t* inputData, const Shape& inputShape, const uint8_t* weights,
                          const Shape& weightsShape, const int32_t* biasData,
                          const Shape& biasShape, int32_t activation, uint8_t* outputData,
                          const Shape& outputShape);

template <typename T>
bool concatenation(const std::vector<const T*>& inputDataPtrs,
                   const std::vector<Shape>& inputShapes, int32_t axis, T* outputData,
                   const Shape& outputShape);

bool l2normFloat16(const _Float16* inputData, const Shape& inputShape, int32_t axis,
                   _Float16* outputData, const Shape& outputShape);
bool l2normFloat32(const float* inputData, const Shape& inputShape, int32_t axis, float* outputData,
                   const Shape& outputShape);
bool localResponseNormFloat16(const _Float16* inputData, const Shape& inputShape, int32_t radius,
                              float bias, float alpha, float beta, int32_t axis,
                              _Float16* outputData, const Shape& outputShape);
bool localResponseNormFloat32(const float* inputData, const Shape& inputShape, int32_t radius,
                              float bias, float alpha, float beta, int32_t axis, float* outputData,
                              const Shape& outputShape);

bool copyData(const void* inputData, const Shape& inputShape, void* outputData,
              const Shape& outputShape);

bool resizeBilinearFloat16(const _Float16* inputData, const Shape& inputShape, _Float16* outputData,
                           const Shape& outputShape);
bool resizeBilinearFloat32(const float* inputData, const Shape& inputShape, float* outputData,
                           const Shape& outputShape);

template <typename T>
bool depthToSpaceGeneric(const T* inputData, const Shape& inputShape, int32_t blockSize,
                         T* outputData, const Shape& outputShape);
template <typename T>
bool spaceToDepthGeneric(const T* inputData, const Shape& inputShape, int32_t blockSize,
                         T* outputData, const Shape& outputShape);

template <typename T>
bool padGeneric(const T* inputData, const Shape& inputShape, const int32_t* paddings, T pad_value,
                T* outputData, const Shape& outputShape);

template <typename T>
bool batchToSpaceGeneric(const T* inputData, const Shape& inputShape, const int32_t* blockSize,
                         T* outputData, const Shape& outputShape);

template <typename T>
bool spaceToBatchGeneric(const T* inputData, const Shape& inputShape, const int32_t* blockSize,
                         const int32_t* padding, const Shape& paddingShape, T* outputData,
                         const Shape& outputShape);

bool subFloat16(const _Float16* in1, const Shape& shape1, const _Float16* in2, const Shape& shape2,
                int32_t activation, _Float16* out, const Shape& shapeOut);

bool subFloat32(const float* in1, const Shape& shape1, const float* in2, const Shape& shape2,
                int32_t activation, float* out, const Shape& shapeOut);

bool subQuant8(const uint8_t* in1, const Shape& shape1, const uint8_t* in2, const Shape& shape2,
               int32_t activation, uint8_t* out, const Shape& shapeOut);

bool divFloat16(const _Float16* in1, const Shape& shape1, const _Float16* in2, const Shape& shape2,
                int32_t activation, _Float16* out, const Shape& shapeOut);
bool divFloat32(const float* in1, const Shape& shape1, const float* in2, const Shape& shape2,
                int32_t activation, float* out, const Shape& shapeOut);

template <typename T>
bool transposeGeneric(const T* inputData, const Shape& inputShape, const int32_t* perm,
                      const Shape& permShape, T* outputData, const Shape& outputShape);

bool meanGeneric(const uint8_t* inputData, const Shape& inputShape, const int32_t* axis,
                 const Shape& axisShape, bool keepDims, uint8_t* outputData,
                 const Shape& outputShape);

bool stridedSliceGeneric(const uint8_t* inputData, const Shape& inputShape,
                         const int32_t* beginData, const int32_t* endData,
                         const int32_t* stridesData, int32_t beginMask, int32_t endMask,
                         int32_t shrinkAxisMask, uint8_t* outputData, const Shape& outputShape);

bool argMinMaxGeneric(const uint8_t* inputData, const Shape& inputShape, int32_t axis,
                      bool isArgMin, uint8_t* outputData, const Shape& outputShape);

bool splitFloat16(const _Float16* inputData, const Shape& inputShape, int32_t axis,
                  const std::vector<_Float16*>* outputDataPtrs,
                  const std::vector<Shape>& outputShapes);

bool splitFloat32(const float* inputData, const Shape& inputShape, const int32_t axis,
                  const std::vector<float*>* outputDataPtrs,
                  const std::vector<Shape>& outputShapes);

bool splitInt32(const int32_t* inputData, const Shape& inputShape, const int32_t axis,
                const std::vector<int32_t*>* outputDataPtrs,
                const std::vector<Shape>& outputShapes);

bool splitQuant8(const uint8_t* inputData, const Shape& inputShape, const int32_t axis,
                 const std::vector<uint8_t*>* outputDataPtrs,
                 const std::vector<Shape>& outputShapes);

bool roiAlignFloat32(const float* inputData, const Shape& inputShape, const float* roiData,
                     const Shape& roiShape, float spatialScale, int32_t samplingRatio,
                     float* outputData, const Shape& outputShape);

bool roiAlignQuant8(const uint8_t* inputData, const Shape& inputShape, const float* roiData,
                    const Shape& roiShape, float spatialScale, int32_t samplingRatio,
                    uint8_t* outputData, const Shape& outputShape);

bool roiPoolingGeneric(const uint8_t* inputData, const Shape& inputShape, const uint8_t* roiData,
                       const Shape& roiShape, float spatialScale, uint8_t* outputData,
                       const Shape& outputShape);

bool heatmapMaxKeypoint(const float* heatmap, const Shape& heatmapShape, const float* boxes,
                        const Shape& boxesShape, float* outputData, const Shape& outputShape);

bool groupedConvFloat32(const float* inputData, const Shape& inputShape, const float* filterData,
                        const Shape& filterShape, const float* biasData, const Shape& biasShape,
                        int32_t numGroups, int32_t padding_left, int32_t padding_right,
                        int32_t padding_top, int32_t padding_bottom, int32_t stride_width,
                        int32_t stride_height, int32_t activation, float* outputData,
                        const Shape& outputShape);

bool groupedConvQuant8(const uint8_t* inputData, const Shape& inputShape, const uint8_t* filterData,
                       const Shape& filterShape, const int32_t* biasData, const Shape& biasShape,
                       int32_t numGroups, int32_t padding_left, int32_t padding_right,
                       int32_t padding_top, int32_t padding_bottom, int32_t stride_width,
                       int32_t stride_height, int32_t activation, uint8_t* outputData,
                       const Shape& outputShape);

bool channelShuffleGeneric(const uint8_t* inputData, const Shape& inputShape, int32_t numGroups,
                           int32_t axis, uint8_t* outputData, const Shape& outputShape);

bool transposeConvFloat32(const float* inputData, const Shape& inputShape, const float* filterData,
                          const Shape& filterShape, const float* biasData, const Shape& biasShape,
                          int32_t padding_left, int32_t padding_right, int32_t padding_top,
                          int32_t padding_bottom, int32_t stride_width, int32_t stride_height,
                          int32_t activation, float* outputData, const Shape& outputShape);

bool transposeConvQuant8(const uint8_t* inputData, const Shape& inputShape,
                         const uint8_t* filterData, const Shape& filterShape,
                         const int32_t* biasData, const Shape& biasShape, int32_t padding_left,
                         int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                         int32_t stride_width, int32_t stride_height, int32_t activation,
                         uint8_t* outputData, const Shape& outputShape);

bool pReluGeneric(const uint8_t* inputData, const Shape& inputShape, const uint8_t* alphaData,
                  const Shape& alphaShape, uint8_t* outputData, const Shape& outputShape);

bool axisAlignedBBoxTransform(const float* roiData, const Shape& roiShape,
                              const float* bboxDeltasData, const Shape& bboxDeltasShape,
                              const float* imageInfoData, const Shape& imageInfoDataShape,
                              const float* weightsData, const Shape& weightsDataShape,
                              bool applyScale, float* outputData, const Shape& outputShape,
                              int32_t* batchSplitData, const Shape& batchSplitShape);

bool rotatedBBoxTransform(const float* roiData, const Shape& roiShape, const float* bboxDeltasData,
                          const Shape& bboxDeltasShape, const float* imageInfoData,
                          const Shape& imageInfoDataShape, const float* weightsData,
                          const Shape& weightsDataShape, bool applyScale, bool angleBoundOn,
                          int32_t angleBoundLow, int32_t angleBoundHigh, float clipAngleThreshold,
                          float* outputData, const Shape& outputShape, int32_t* batchSplitData,
                          const Shape& batchSplitShape);
}  // namespace nn
}  // namespace android
#endif  // ANDROID_ML_NN_COMMON_OPERATIONS_H
