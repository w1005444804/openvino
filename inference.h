#ifndef INFERENCE_API_H_
#define INFERENCE_API_H_
#pragma once
#include <stdio.h>
#include <string>
#include <map>

#ifdef _MSC_VER
#ifdef INFERENCE_EXPORTS
#define INFERENCE_API __declspec(dllexport)
#else
#define INFERENCE_API __declspec(dllimport)
#endif
#else
#define INFERENCE_API
#endif

struct Shape {
	unsigned int N{ 0 };
	unsigned int C{ 0 };
	unsigned int H{ 0 };
	unsigned int W{ 0 };
	unsigned int count() {
		return N*C*H*W;
	}
};

struct IFparams {
	char* Root_Path;
	char* Model_Path;
	char* Proto_Path;
	bool CPU_ONLY{ 1 };
	//Shape shape;
};

struct Datum {
	Shape shape;
	float* outter_data = nullptr;
	float* data = nullptr;//NCHW

	Datum() { 
		data = nullptr;
	}
	float* Getdata() {
		if (data != nullptr) {
			return data;
		}
		else {
			unsigned int size = shape.count();
			data = new float[size];
			return data;
		}
	}
	void Reshape(Shape s) {
		shape = s;
		if (data != nullptr) {
			delete[] data;
			data = nullptr;
		}
	}
	~Datum() {
		if (data != nullptr){
			delete[] data;
			data = nullptr;
		}
	}
};
static bool compare_shape(const Shape& S1, const Shape& S2) {
	if (S1.W != S2.W || S1.H != S2.H ||
		S1.C != S2.C || S1.N != S2.N) {
		return 1;
	}
	return 0;
}
class RPNDetection;

class INFERENCE_API TNT {
	friend class RPNDetection;
public:

	typedef std::map<std::string, Datum> IFReult;

	explicit TNT(const IFparams& ifp);

	virtual~TNT();

	void Init(const IFparams& ifp);

	void Inference(const Datum& Data);

	TNT::IFReult* GetReult();

private:
	void* Engine;

	IFReult* OutPut_;
};
#endif