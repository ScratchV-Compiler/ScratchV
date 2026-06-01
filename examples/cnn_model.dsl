; ============================================================
; Auto-generated ScratchV DSL from cnn.onnx
; 15 operators, 12 weights
; ============================================================

_layer1_0_Conv_output_0 = conv(input, layer1_0_weight, layer1_0_bias, out_channels:32, kernel_size:3, stride:1, padding:0)
_layer1_1_Relu_output_0 = relu(_layer1_0_Conv_output_0)
_layer1_2_MaxPool_output_0 = maxpool(_layer1_1_Relu_output_0, kernel:2, stride:2)
_layer2_0_Conv_output_0 = conv(_layer1_2_MaxPool_output_0, layer2_0_weight, layer2_0_bias, out_channels:32, kernel_size:3, stride:1, padding:0)
_layer2_1_Relu_output_0 = relu(_layer2_0_Conv_output_0)
_layer2_2_MaxPool_output_0 = maxpool(_layer2_1_Relu_output_0, kernel:2, stride:2)
_layer3_0_Conv_output_0 = conv(_layer2_2_MaxPool_output_0, layer3_0_weight, layer3_0_bias, out_channels:64, kernel_size:3, stride:1, padding:0)
_layer3_1_Relu_output_0 = relu(_layer3_0_Conv_output_0)
_layer3_2_MaxPool_output_0 = maxpool(_layer3_1_Relu_output_0, kernel:2, stride:2)
PPQ_Variable_12 = reshape(_layer3_2_MaxPool_output_0, PPQ_Variable_13)
_fc1_Gemm_output_0 = gemm(PPQ_Variable_12, fc1_weight, fc1_bias, transA:0, transB:1)
_relu1_Relu_output_0 = relu(_fc1_Gemm_output_0)
_fc2_Gemm_output_0 = gemm(_relu1_Relu_output_0, fc2_weight, fc2_bias, transA:0, transB:1)
output1 = sigmoid(_fc2_Gemm_output_0)
PPQ_Variable_24 = reshape(output1, PPQ_Variable_25)

return PPQ_Variable_24
