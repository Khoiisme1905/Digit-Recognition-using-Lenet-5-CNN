#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#define L2_LAMBDA 0.0001
#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y] - L2_LAMBDA * weight[x][y]; \
}
static double previous_deltas[sizeof(LeNet5) / sizeof(double)] = {0};

// Thêm vào lenet.c
static double learning_rate = ALPHA;
static double momentum_rate = MOMENTUM;

void update_learning_rate(double decay_rate) {
    learning_rate *= decay_rate;
    printf("Learning rate updated to: %.5f\n", learning_rate);
}
static inline void load_input(Feature *features, image input)
{
    double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
            // Chuẩn hóa dữ liệu tốt hơn
            layer0[0][j + PADDING][k + PADDING] = (input[j][k] / 255.0) * 2.0 - 1.0; // Chuẩn hóa về [-1, 1]
        }
    }
}
// Thêm vào lenet.c


double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	batch_normalize((double *)features->layer1, LAYER1 * LENGTH_FEATURE1 * LENGTH_FEATURE1, 1e-5);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	batch_normalize((double *)features->layer3, LAYER3 * LENGTH_FEATURE3 * LENGTH_FEATURE3, 1e-5);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	batch_normalize((double *)features->layer5, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, 1e-5);
	apply_dropout((double *)features->layer5, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, DROPOUT_RATE);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}



static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
    double max_val = input[0];
    for (int i = 1; i < count; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    double sum_exp = 0.0;
    for (int i = 0; i < count; i++) {
        loss[i] = exp(input[i] - max_val); // Tránh tràn số bằng cách trừ đi max_val
        sum_exp += loss[i];
    }

    for (int i = 0; i < count; i++) {
        loss[i] /= sum_exp; // Chuẩn hóa thành xác suất
    }

    // Tính lỗi cho backpropagation
    for (int i = 0; i < count; i++) {
        loss[i] = (i == label) - loss[i];
    }
}

inline void apply_dropout(double *layer, int size, double rate) {
    for (int i = 0; i < size; i++) {
        if ((double)rand() / RAND_MAX < rate) {
            layer[i] = 0;
        } else {
            layer[i] /= (1.0 - rate); // Scaling để giữ giá trị kỳ vọng
        }
    }
}

// Thêm hàm batch normalization
void batch_normalize(double *layer, int size, double epsilon) {
	// Tính mean
    double mean = 0.0;
    for (int i = 0; i < size; i++) {
        mean += layer[i];
    }
    mean /= size;
    
    // Tính variance
    double var = 0.0;
    for (int i = 0; i < size; i++) {
        var += (layer[i] - mean) * (layer[i] - mean);
    }
    var /= size;
    
    // Normalize
    for (int i = 0; i < size; i++) {
        layer[i] = (layer[i] - mean) / sqrt(var + epsilon);
    }
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}

void augment_image(image input, image output) {
    // Dịch ảnh ngẫu nhiên
    int shift_x = rand() % 3 - 1;
    int shift_y = rand() % 3 - 1;
    
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            int new_i = i + shift_x;
            int new_j = j + shift_y;
            
            // Đảm bảo không vượt ra ngoài biên
            if (new_i >= 0 && new_i < 28 && new_j >= 0 && new_j < 28) {
                output[i][j] = input[new_i][new_j];
            } else {
                output[i][j] = 0;
            }
        }
    }
}

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = learning_rate / batchSize;
    FOREACH(i, GETCOUNT(LeNet5)) {
        double delta = k * buffer[i] + momentum_rate * previous_deltas[i];
        ((double *)lenet)[i] += delta;
        previous_deltas[i] = delta;
	}
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input, uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    forward(lenet, &features, relu);

    double *output = features.output;
    uint8 result = 0;
    double max_value = output[0];
    for (uint8 i = 1; i < count; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            result = i;
        }
    }
    return result;
}
uint8 EnsemblePredict(LeNet5 **models, int num_models, image input, uint8 count)
{
    double votes[OUTPUT] = {0}; // OUTPUT là 10 cho MNIST
    
    for (int m = 0; m < num_models; m++) {
        Feature features = {0};
        load_input(&features, input);
        forward(models[m], &features, relu);
        
        // Lấy các xác suất từ mô hình
        double probs[OUTPUT];
        double sum_exp = 0.0;
        double max_val = features.output[0];
        
        // Tìm max để tránh overflow
        for (uint8 i = 1; i < count; i++) {
            if (features.output[i] > max_val) max_val = features.output[i];
        }
        
        // Tính softmax
        for (uint8 i = 0; i < count; i++) {
            probs[i] = exp(features.output[i] - max_val);
            sum_exp += probs[i];
        }
        for (uint8 i = 0; i < count; i++) {
            probs[i] /= sum_exp;
            votes[i] += probs[i]; // Cộng vào tổng votes
        }
    }
    
    // Tìm class có số vote cao nhất
    uint8 result = 0;
    double max_vote = votes[0];
    for (uint8 i = 1; i < count; i++) {
        if (votes[i] > max_vote) {
            max_vote = votes[i];
            result = i;
        }
    }
    
    return result;
}
void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"