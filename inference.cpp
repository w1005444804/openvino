#include <gflags/gflags.h>
#include <functional>
#include <chrono>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>
#include <opencv2/opencv.hpp>
#include "inference.h"
#include "pend.h"
#include <samples/slog.hpp>
#pragma once
#include <fstream>

using namespace InferenceEngine;

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

// --------------------------- 1. Load Plugin for IF engine -------------------------------------
InferencePlugin plugin;
//----------------------------- 2. Read IR models and load them to plugins ------------------------------
CNNNetReader network_reader;

CNNNetwork network;

ExecutableNetwork executable_network;
//-----------------------------------3. Configure input & output--------------------------------
/** Taking information about all topology inputs **/
InferenceEngine::InputsDataMap input_info;

/** Taking information about all topology outputs **/
InferenceEngine::OutputsDataMap output_info;

/** Iterating over all input info**/
/** One net has only one input blob*/
InputInfo::Ptr input_data;

//------------------------------------------------

void IF::Init(const IFparams& ifp) {
	//* Load  network. */
	{
		// --------------------------- 1. Load Plugin for inference engine -------------------------------------
		std::string DeviceName;
		if (ifp.CPU_ONLY){
			DeviceName = "CPU";
		}
		else {
			DeviceName = "GPU";
		}
		try { 
			plugin = PluginDispatcher({ "../x64","" }).getPluginByDevice(DeviceName); 
		}
		catch (const std::exception& error) {
			std::cerr << "[ ERROR ] " << error.what() << std::endl;
			system("pause");
		}
		catch (...) {
			std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
			system("pause");
		}
		if (ifp.CPU_ONLY) {
			plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
		}
		printPluginVersion(plugin, std::cout);
		//----------------------------- 2. Read IR models  ------------------------------
		network_reader.ReadNetwork(ifp.Proto_Path);
		network_reader.ReadWeights(ifp.Model_Path);
		//-----------------------------------3. Configure input--------------------------------
		network = network_reader.getNetwork();

		network.setBatchSize(1);
		/** Taking information about all topology inputs **/
		input_info = network.getInputsInfo();
		/** Taking information about all topology outputs **/
		output_info = network.getOutputsInfo();

		/** Iterating over all input info**/
		/** One net has only one input blob*/
		input_data = input_info.begin()->second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		//--------------------------------- 4. Load them to plugins --------------------------
		executable_network = plugin.LoadNetwork(network, {});
	}
}

void IF::ConfigureOut() {

}

void IF::ReshapeNet(const Shape& input_shape) {
	if (input_shape.W != Input_Shape_.W || input_shape.H != Input_Shape_.H ||
		input_shape.C != Input_Shape_.C || input_shape.H != Input_Shape_.H){
		SizeVector input_dims = input_data->getTensorDesc().getDims();
		input_dims[0] = input_shape.N;
		input_dims[1] = input_shape.C;
		input_dims[2] = input_shape.H;
		input_dims[3] = input_shape.W;
		std::map<std::string, SizeVector> input_shapes;
		input_shapes[input_info.begin()->first] = input_dims;
		network.reshape(input_shapes);
		executable_network = plugin.LoadNetwork(network, {});
	}
}

void IF::inference(Datum& Data) {
	//* -------------end refine network and start classify network----------------------*//
	ReshapeNet(Data.shape);

	InferRequest infer_request = executable_network.CreateInferRequest();
	Blob::Ptr inputblobptr = infer_request.GetBlob(input_data->name());
	float* data = inputblobptr->buffer().as<float*>();
	std::memcpy(data, Data.data, inputblobptr->size());
	auto t0 = std::chrono::high_resolution_clock::now();
	infer_request.StartAsync();
	infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);
	auto t1 = std::chrono::high_resolution_clock::now();
	ms detection = std::chrono::duration_cast<ms>(t1 - t0);
	
	std::cout << "detection time  : " << std::fixed << std::setprecision(2) << detection.count()
		<< " ms (" << 1000.f / detection.count() << " fps)" << std::endl;
	/*-----------------------------------get output blob--------------------------------------------------------------*/
	std::map<std::string, Datum> OutPutBlob;

	for (auto it = output_info.begin(); it != output_info.end(); ++it){
		std::string name = it->first;
		Blob::Ptr blobptr = infer_request.GetBlob(name);
		const unsigned int size = blobptr->size();
		float* data = blobptr->buffer().as<float*>();

		DataPtr dataptr = it->second;
		SizeVector SV = dataptr->getTensorDesc().getDims();

		Datum datum;
		datum.shape.N = SV[0];
		datum.shape.C = SV[1];
		datum.shape.H = SV[2];
		datum.shape.W = SV[3];
		
		datum.data = new float[size];
		std::memcpy(datum.data, data, size);

		std::pair<std::string, Datum> name_data;
		OutPutBlob.insert(std::make_pair(name, datum));
	}
}


IF::IFReult* IF::GetReult() {
	return &OutPut_;
}