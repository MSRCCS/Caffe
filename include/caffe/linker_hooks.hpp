// Layers are typically instantiated using LayerRegistry::CreateLayer() factory
// method (see layer_factory.hpp). This mechanism assumes that all object files
// (even those which don't contain any referenced symbols) get linked into the
// final executable. In GCC this behavior is forced by passing --whole-archive
// to the linker. Unfortunately, Visual Studio linker doesn't have a similar
// option. To work around that, a dummy variable ("hook") is defined in layer's
// .cpp file through REGISTER_LAYER_CREATOR/ENSURE_HOOKED macro and then used
// in net.cpp through USE_HOOK macro. This forces linking the entire .obj file
// for the layer.
// 
// The same logic applies to solvers.

#pragma once

#if defined(_MSC_VER)

namespace caffe {

#define ENSURE_HOOKED(X) X##Class_hook X##Class_hook_instance;

  // Macro that declares external members that can get used as a hook to ensure that OBJ file will
  // get linked into final binary. Typedef is used only to ensure that all classes will 
  // be hooked (otherwise there will be compiler error).
#define CREATE_HOOK(X) \
  typedef int X##Class_hook; \
  extern X##Class_hook X##Class_hook_instance

  // DummyFunction is used to ensure that the external symbol gets used in executable,
  // to ensure that object file where symbol is defined will get linked.
  template<class T>
  void DummyFunction(T) {}

  // Macro used to ensure that all external symbols get used in executable, ensuring that
  // their object files will get linked into resulting binary.
#define USE_HOOK(X) caffe::DummyFunction(caffe::X##Class_hook_instance); \

  // Macro that allows you to use another macro on all layer types.
#define FOR_ALL_STANDARD_LAYERS(FUNC) \
  FUNC(AbsVal); \
  FUNC(Accuracy); \
  FUNC(ArgMax); \
  FUNC(BatchNorm); \
  FUNC(BatchReindex); \
  FUNC(Bias); \
  FUNC(BNLL); \
  FUNC(CCALoss); \
  FUNC(CenterLoss); \
  FUNC(Concat); \
  FUNC(ContrastiveLoss); \
  FUNC(Convolution); \
  FUNC(Crop); \
  FUNC(CrossEntropyLoss); \
  FUNC(Deconvolution); \
  FUNC(DenseLoss); \
  FUNC(DetectionEvaluate); \
  FUNC(DetectionOutput); \
  FUNC(Dropout); \
  FUNC(DummyData); \
  FUNC(Eltwise); \
  FUNC(ELU); \
  FUNC(EuclideanLoss); \
  FUNC(IndexedThresholdLoss); \
  FUNC(Exp); \
  FUNC(Embed); \
  FUNC(Filter); \
  FUNC(Flatten); \
  FUNC(Data);    \
  FUNC(HDF5Data); \
  FUNC(HDF5Output); \
  FUNC(HingeLoss); \
  FUNC(Im2col); \
  FUNC(ImageData); \
  FUNC(InfogainLoss); \
  FUNC(InnerProduct); \
  FUNC(Input); \
  FUNC(L2Norm); \
  FUNC(Log); \
  FUNC(LRN); \
  FUNC(LSTM); \
  FUNC(LSTMUnit); \
  FUNC(MemoryData); \
  FUNC(MIL); \
  FUNC(MILData); \
  FUNC(MultinomialLogisticLoss); \
  FUNC(MultiAccuracy); \
  FUNC(MultiBoxLoss); \
  FUNC(MVN); \
  FUNC(Normalize); \
  FUNC(Parameter); \
  FUNC(Permute); \
  FUNC(Pooling); \
  FUNC(Power); \
  FUNC(PReLU); \
  FUNC(PriorBox); \
  FUNC(Python); \
  FUNC(Reduction); \
  FUNC(RegionLoss); \
  FUNC(RegionOutput); \
  FUNC(RegionPrediction); \
  FUNC(YoloBBs); \
  FUNC(YoloEvalCompat); \
  FUNC(RegionTarget); \
  FUNC(ReLU); \
  FUNC(Reorg); \
  FUNC(Reshape); \
  FUNC(RNN); \
  FUNC(ROIPooling); \
  FUNC(RPNProposal); \
  FUNC(Scale); \
  FUNC(SgmLoss); \
  FUNC(Resize); \
  FUNC(Sigmoid); \
  FUNC(Silence); \
  FUNC(Slice); \
  FUNC(SigmoidCrossEntropyLoss); \
  FUNC(SmoothL1Loss); \
  FUNC(Softmax); \
  FUNC(SoftmaxWithLoss); \
  FUNC(SoftmaxTree); \
  FUNC(SoftmaxTreeWithLoss); \
  FUNC(TreePrediction); \
  FUNC(SoftmaxTreePrediction); \
  FUNC(NMSFilter); \
  FUNC(YoloCoOccurrence); \
  FUNC(Split); \
  FUNC(SPP); \
  FUNC(TanH); \
  FUNC(Threshold); \
  FUNC(Tile); \
  FUNC(TripletLoss); \
  FUNC(TsvBoxData); \
  FUNC(TsvCPMData); \
  FUNC(TsvData); \
  FUNC(WSgmLoss); \
  FUNC(WindowData); \
  FUNC(XCovLoss); \
  FUNC(Axpy)

#ifdef WITH_PYTHON_LAYER
#define FOR_PYTHON_LAYER(FUNC) \
  FUNC(Python)
#else
#define FOR_PYTHON_LAYER(FUNC)
#endif

#define FOR_ALL_LAYERS(FUNC) \
  FOR_ALL_STANDARD_LAYERS(FUNC); \
  FOR_PYTHON_LAYER(FUNC)

#define FOR_ALL_SOLVERS(FUNC) \
  FUNC(AdaDelta); \
  FUNC(AdaGrad); \
  FUNC(Adam); \
  FUNC(Nesterov); \
  FUNC(RMSProp); \
  FUNC(SGD)

  FOR_ALL_LAYERS(CREATE_HOOK);
  FOR_ALL_SOLVERS(CREATE_HOOK);
}
#else
#define FOR_ALL_LAYERS(FUNC)
#define ENSURE_HOOKED(X)
#endif
