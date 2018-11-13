#pragma once
#include <gflags/gflags.h>
#include <functional>
#include <chrono>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>
#include <samples/slog.hpp>

#include "inference.h"

using namespace InferenceEngine;

class openvinoinfer {
public:
	explicit openvinoinfer (const IFparams& ifparam);

	~openvinoinfer ();

	void Init(const IFparams& ifp);

	void inference(const Datum& Data);

	inline void ReshapeNet(const Shape& Input_shape);

	TNT::IFReult& GetReult();

	friend TNT;

public:

	void ConfigureOut();

	Shape Getshape();

private:
	Shape Input_Shape;

	TNT::IFReult OutPut;

private:
	// --------------------------- 1. Load Plugin for IF engine -------------------------------------
	InferencePlugin plugin;
	//----------------------------- 2. Read IR models and load them to plugins ------------------------------
	CNNNetReader network_reader;

	CNNNetwork network;

	ExecutableNetwork executable_network;

	InferRequest infer_request;
	//-----------------------------------3. Configure input & output--------------------------------
	/** Taking information about all topology inputs **/
	InferenceEngine::InputsDataMap input_info;

	/** Taking information about all topology outputs **/
	InferenceEngine::OutputsDataMap output_info;

	/** Iterating over all input info**/
	/** One net has only one input blob*/
	InputInfo::Ptr input_data;
};

