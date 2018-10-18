#ifndef INFERENCE_API_H_
#define INFERENCE_API_H_

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

struct IFparams {
	char* Model_Path;
	char* Proto_Path;
	bool CPU_ONLY{ 1 };
};

struct Shape {
	unsigned int N{0};
	unsigned int C{0};
	unsigned int H{0};
	unsigned int W{0};
};

struct Datum {
	Shape shape;
	void* data;//NCHW
	Datum() { 
		data = nullptr;
	}
	~Datum() {
		delete[] data;
	}
};


class INFERENCE_API IF {
public:

	typedef std::map<std::string, Datum> IFReult;

	IF() {}

	virtual~IF() {}

	void Init(const IFparams& ifp);

	void inference(Datum& Data);

	IFReult* GetReult();

private:

	void ReshapeNet(const Shape& Input_shape);

	void ConfigureOut();

private:

	Shape Input_Shape_;

	IFReult OutPut_;
};
#endif