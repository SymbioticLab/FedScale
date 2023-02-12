#!bin/bash
cd ../../cloud/aggregation/cache

# convert model: onnx->mnn->json.
../../../scripts/MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --forTraining --bizCode fedscale
../../../scripts/MNNDump2Json model.mnn model.json 
