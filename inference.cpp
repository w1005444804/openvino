
//#include <opencv2/opencv.hpp>
#include "inference.h"
#include "openvino.h"
#pragma once
#include <fstream>

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;


//-----------------------------------------------
openvinoinfer::openvinoinfer(const IFparams& param) {
	Init(param);
}
void openvinoinfer::Init(const IFparams& param) {
	// --------------------------- 1. Load Plugin for inference engine -------------------------------------
	std::string DeviceName;
	if (param.CPU_ONLY) {
		DeviceName = "CPU";
	} else {
		DeviceName = "GPU";
	}
	try {
		plugin = PluginDispatcher({ param.Root_Path,"./","" }).getPluginByDevice(DeviceName);
	}
	catch (const std::exception& error) {
		std::cerr << "[ ERROR ] " << error.what() << std::endl;
		system("pause");
	}
	catch (...) {
		std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
		system("pause");
	}
	if (param.CPU_ONLY) {
		plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
	}
	printPluginVersion(plugin, std::cout);
	//----------------------------- 2. Read IR models  ------------------------------
	network_reader.ReadNetwork(param.Proto_Path);
	network_reader.ReadWeights(param.Model_Path);
	//-----------------------------------3. Configure input--------------------------------
	network = network_reader.getNetwork();

	//network.setBatchSize(1);
	/** Taking information about all topology inputs **/
	input_info = network.getInputsInfo();
	/** Taking information about all topology outputs **/
	output_info = network.getOutputsInfo();

	/** Iterating over all input info**/
	/** One net has only one input blob*/
	input_data = input_info.begin()->second;
	input_data->setPrecision(Precision::FP32);
	input_data->setLayout(Layout::NCHW);
	auto shape = input_data->getTensorDesc().getDims();
	Input_Shape.N = shape[0];
	Input_Shape.C = shape[1];
	Input_Shape.H = shape[2];
	Input_Shape.W = shape[3];

	//--------------------------------- 4. Load them to plugins --------------------------
	executable_network = plugin.LoadNetwork(network, {});
	infer_request = executable_network.CreateInferRequest();
	//-----------------------------------3. Configure output--------------------------------
	ConfigureOut();
	
}

void openvinoinfer::ConfigureOut() {
	for (auto it = output_info.begin(); it != output_info.end(); ++it) {
		std::string name = it->first;
		Blob::Ptr blobptr = infer_request.GetBlob(name);
		const unsigned int size = blobptr->size();
		float* data = blobptr->buffer().as<float*>();

		DataPtr dataptr = it->second;
		SizeVector SV = dataptr->getTensorDesc().getDims();

		if (OutPut.count(name) == 0) {
			Datum datum;
			OutPut.insert(std::make_pair(name, datum));
		}
		auto outit = OutPut.find(name);

		Datum& datum = outit->second;

		datum.shape.N = SV[0];
		datum.shape.C = SV[1];
		datum.shape.H = SV[2];
		datum.shape.W = SV[3];

		/*if (datum.data != nullptr) {
			delete[] datum.data;
		}
		datum.data = new float[size];*/
	}
}


Shape openvinoinfer::Getshape() {
	return Input_Shape;
}

void openvinoinfer::ReshapeNet(const Shape& input_shape) {
	if (compare_shape(input_shape, Input_Shape)) {
		Input_Shape = input_shape;
		SizeVector input_dims = input_data->getTensorDesc().getDims();
		input_dims[0] = input_shape.N;
		input_dims[1] = input_shape.C;
		input_dims[2] = input_shape.H;
		input_dims[3] = input_shape.W;
		std::map<std::string, SizeVector> input_shapes;
		input_shapes[input_info.begin()->first] = input_dims;
		network.reshape(input_shapes);
		executable_network = plugin.LoadNetwork(network, {});
		infer_request = executable_network.CreateInferRequest();

		ConfigureOut();
	}
}

void openvinoinfer::inference(const Datum& Data) {
	//* -------------end refine network and start classify network----------------------*//
	ReshapeNet(Data.shape);
	/*------------------feed data into input_data------------------------------*/
	Blob::Ptr inputblobptr = infer_request.GetBlob(input_data->name());
	float* data = inputblobptr->buffer().as<float*>();
	std::memcpy(data, Data.data, inputblobptr->size() * sizeof(float));
	/*----------------------Forward-----------------------------*/
	auto t0 = std::chrono::high_resolution_clock::now();
	infer_request.StartAsync();
	infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);

	auto t1 = std::chrono::high_resolution_clock::now();
	ms detection = std::chrono::duration_cast<ms>(t1 - t0);
	std::cout << "detection time  : " << std::fixed << std::setprecision(2) << detection.count()
		<< " ms (" << 1000.f / detection.count() << " fps)" << std::endl;

	/*-----------------------------------get output blob--------------------------------------------------------------*/
	for (auto it = output_info.begin(); it != output_info.end(); ++it) {
		std::string name = it->first;
		Blob::Ptr blobptr = infer_request.GetBlob(name);
		const unsigned int size = blobptr->size();
		float* data = blobptr->buffer().as<float*>();

		DataPtr dataptr = it->second;
		SizeVector SV = dataptr->getTensorDesc().getDims();
		assert(OutPut.count(name) != 0);
		Datum& datum = OutPut.find(name)->second;
		datum.shape.N = SV[0];
		datum.shape.C = SV[1];
		datum.shape.H = SV[2];
		datum.shape.W = SV[3];

		datum.outter_data = data;
		//std::memcpy((void*)datum.data, (void*)data, size * sizeof(float));
	}
}

TNT::IFReult& openvinoinfer::GetReult() {
	return OutPut;
}

openvinoinfer ::~openvinoinfer(){

}
//------------------------------------------------
//------------------------------------------------
//----------------API CLASS TNT-------------------
//------------------------------------------------
//------------------------------------------------
TNT::TNT(const IFparams& ifp) {
	Init(ifp);
}
void TNT::Init(const IFparams& ifp) {
	openvinoinfer* vinoptr = new openvinoinfer(ifp);
	Engine = static_cast<void*>(vinoptr);
}

void TNT::Inference(const Datum& Data) {
	static_cast<openvinoinfer*>(Engine)->inference(Data);
}

TNT::IFReult* TNT::GetReult() {
	return &(static_cast<openvinoinfer*>(Engine)->GetReult());
}

TNT::~TNT(){
	delete static_cast<openvinoinfer*>(Engine);
}