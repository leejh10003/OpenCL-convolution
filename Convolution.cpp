#include "OpenCL-platform-Initializer/jjOpenCLPlatformInitialize.hpp"
#define DIMENSION 1
const unsigned int inputSignalWidth = 8;
const unsigned int inputSignalHeight = 8;
using namespace std;
cl_uint inputSignal [inputSignalWidth][inputSignalHeight] = 
{
	{3,1,1,4,8,2,1,3},
	{4,2,1,1,2,1,2,3},
	{4,4,4,4,3,2,2,2},
	{9,8,3,8,9,0,0,0},
	{9,3,3,9,0,0,0,0},
	{0,9,0,8,0,0,0,0},
	{3,0,8,8,9,4,4,4},
	{5,9,8,1,8,1,1,1}
};

const unsigned int outputSignalWidth = 6;
const unsigned int outputSignalHeight = 6;

cl_uint outputSignal[outputSignalWidth][outputSignalHeight];

const unsigned int maskWidth = 3;
const unsigned int maskHeight = 3;
cl_uint mask [maskWidth][maskHeight] = 
{
	{1,1,1},
	{1,0,1},
	{1,1,1}
};

inline void checkErr(
	cl_int err,
	const char* name)
{
	if(err != CL_SUCCESS)
	{
		cerr << "ERROR: " << name
			 << " (" << err << ")" << endl;
		exit(EXIT_FAILURE);
	}
}

void CL_CALLBACK contextCallback(
	const char* errInfo,
	const void* private_info,
	size_t cb,
	void* user_data)
{
	cerr << "Error occurred during context use: "
		 << errInfo << endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
	cl_int errNum;
	JJ_CL_PLATFORMS platformStr;
	cl_uint numDevices = NULL;
	cl_device_id* deviceIDs = NULL;
	cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	errNum = jjOpenCLPlatformInitialize(&platformStr, true);
	cl_uint i;
	for(i = 0; i < platformStr.platformsNum; ++i)
	{
		errNum = clGetDeviceIDs(
			platformStr.platforms[i].platformID,
			CL_DEVICE_TYPE_CPU, 
			0, NULL,
			&numDevices);
		if(errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
		{
			checkErr(errNum, "clGetDeviceIDs");
		}
		else if(numDevices > 0)
		{
			deviceIDs = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
			platformStr.platforms[i].platformID,
			CL_DEVICE_TYPE_CPU,
			numDevices, deviceIDs,
			NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
		}
	}
	if(deviceIDs == NULL)
	{
		cout << "No CPU device found" <<endl;
		exit(-1);
	}
	cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties)platformStr.platforms[0].platformID, 0
	};
	context = clCreateContext(
		contextProperties, numDevices, deviceIDs,
		contextCallback, NULL, &errNum);
	checkErr(errNum, "clCreateContext");

	ifstream srcFile("Convolution.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");
	string srcProg(istreambuf_iterator<char>(srcFile), (istreambuf_iterator<char>()));
	const char* src = srcProg.c_str();
	size_t length = srcProg.length();

	program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	errNum = clBuildProgram(program, numDevices, deviceIDs, NULL, NULL, NULL);
	checkErr(errNum, "clBuildProgram");

	kernel = clCreateKernel(program, "convolve", &errNum);
	checkErr(errNum, "clCreateKernel");

	inputSignalBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(cl_uint) * inputSignalHeight * inputSignalWidth, static_cast<void*>(inputSignal), &errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(cl_uint) * maskWidth * maskHeight, static_cast<void*>(mask), &errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
	sizeof(cl_uint) * outputSignalHeight * outputSignalWidth, 0, &errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	queue = clCreateCommandQueue(context, deviceIDs[0], NULL, &errNum);
	checkErr(errNum, "clCreateCommandQueue");

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[DIMENSION] = {outputSignalWidth * outputSignalHeight};
	const size_t localWorkSize[DIMENSION] = {1};

	errNum = clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	checkErr(errNum, "clEnqueueNDRangeKernrl");

	errNum = clEnqueueReadBuffer(queue, outputSignalBuffer, CL_TRUE, 0, sizeof(cl_uint) * outputSignalHeight * outputSignalWidth, outputSignal, 0, NULL, NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
	for(int y = 0; y < outputSignalHeight; ++y){
		for(int x = 0; x < outputSignalWidth; ++x)
		{
			cout << outputSignal[x][y] << " ";
		}
		cout << endl;
	}
	return 0;
}
