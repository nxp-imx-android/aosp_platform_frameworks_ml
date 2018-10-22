// clang-format off
// Generated file (from: mean_quant8_1.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type0(Type::TENSOR_QUANT8_ASYMM, {4, 3, 2}, 0.8f, 5);
  OperandType type1(Type::TENSOR_INT32, {4});
  OperandType type2(Type::INT32, {});
  OperandType type3(Type::TENSOR_QUANT8_ASYMM, {2}, 0.8f, 5);
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto axis = model->addOperand(&type1);
  auto keepDims = model->addOperand(&type2);
  auto output = model->addOperand(&type3);
  // Phase 2, operations
  static int32_t axis_init[] = {1, 0, -3, -3};
  model->setOperandValue(axis, axis_init, sizeof(int32_t) * 4);
  static int32_t keepDims_init[] = {0};
  model->setOperandValue(keepDims, keepDims_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_MEAN, {input, axis, keepDims}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {input},
    {output});
  assert(model->isValid());
}

inline bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}

