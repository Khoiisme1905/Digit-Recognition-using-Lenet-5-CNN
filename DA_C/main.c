/*
//  gcc -I ./ main.c lenet.c -lm -o main
//  emcc -s ALLOW_MEMORY_GROWTH=1 --preload-file ./ -I ./ main.c lenet.c -lm -o // --preload-file is for web, we need --embed-file
// https://github.com/emscripten-core/emscripten/issues/4756
// emcc -s ALLOW_MEMORY_GROWTH=1 --embed-file ./ -I ./ main.c lenet.c -lm -o   // we dont need to embed the whole dir, this will casue the a.out.js embed the whole dir, 
// 																			// a.out.js will be very big, and program will crash when run, so just embed test data and NN model
// emcc -s ALLOW_MEMORY_GROWTH=1 --embed-file t10k-images-idx3-ubyte --embed-file t10k-labels-idx1-ubyte --embed-file model.dat -I ./ main.c lenet.c -lm -o

// use NODERAWFS, becasue your a.out.js will be very large (include the whole embed-file) if you use embed-file
// emcc -s ALLOW_MEMORY_GROWTH=1 -s NODERAWFS=1 -I ./ lenet.c main.c -lm -o
// wasm32-wasi-clang   -I ./ lenet.c main.c -lm -o mainWasi.wasm
// wasmtime --dir=. mainWasi.wasm   // capability safety by --dir=. give permission to access file system.
*/

#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define INPUT_FILE "input.idx3-ubyte"
#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000
// #define COUNT_TEST		5

// Thêm vào main.c
void train_with_early_stopping(LeNet5 *lenet, image *train_data, uint8 *train_label, 
                               image *valid_data, uint8 *valid_label,
                               int batch_size, int max_epochs, int patience)
{
    double best_accuracy = 0.0;
    int no_improvement = 0;
    
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // Training một epoch
        training(lenet, train_data, train_label, batch_size, COUNT_TRAIN);
        
        // Tính accuracy trên tập validation (có thể sử dụng một phần của tập test)
        int right = testing(lenet, valid_data, valid_label, COUNT_TEST / 5);
        double accuracy = (double)right / (COUNT_TEST / 5);
        
        printf("Epoch %d: validation accuracy = %.2f%%\n", epoch, accuracy * 100);
        
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            no_improvement = 0;
            // Lưu mô hình tốt nhất
            save(lenet, 0, accuracy, "best_model.dat");
        } else {
            no_improvement++;
        }
        
        if (no_improvement >= patience) {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }
    }
    
    // Load lại mô hình tốt nhất
    load(lenet, "best_model.dat");
}
int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
				printf("Inside testing, before predict, we are OK\n");

		// int p = Predict(lenet, test_data[i], 10);
		int p = Predict(lenet, test_data[i], 10);

		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}

int save(LeNet5 *lenet, double train_accuracy, double test_accuracy, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(&train_accuracy, sizeof(double), 1, fp);
    fwrite(&test_accuracy, sizeof(double), 1, fp);
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    double train_acc, test_acc;
    fread(&train_acc, sizeof(double), 1, fp);
    fread(&test_acc, sizeof(double), 1, fp);
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    printf("Loaded model with training accuracy: %.2f%%, test accuracy: %.2f%%\n", 
           train_acc * 100, test_acc * 100);
    return 0;
}



void foo()
{
	
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

	// * commnet train part becasue we use a pre trained model
	// if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	// {
	// 	printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
	// 	free(train_data);
	// 	free(train_label);
	// 	// system("pause");
	// 	exit(1);
	// }
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		// system("pause");
		exit(1);

	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	clock_t start = clock();

	// * comment training, use pre-trained models
	// int batches[] = { 300 };
	// for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
	// 	training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);

	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)(clock() - start));
	double train_accuracy = 0.0; // Tính từ kết quả training
	double test_accuracy = (double)right / COUNT_TEST;
	save(lenet, train_accuracy, test_accuracy, LENET_FILE);
	free(lenet);
	// free(train_data);
	// free(train_label);
	free(test_data);
	free(test_label);
	// system("pause");
	// Tách 1/5 dữ liệu test để làm validation
	image *valid_data = test_data;
	uint8 *valid_label = test_label;
	int valid_size = COUNT_TEST / 5;

	// Training với early stopping
	train_with_early_stopping(lenet, train_data, train_label, 
                         valid_data, valid_label, 
                         100, 30, 5); // batch_size=100, max_epochs=30, patience=5
	exit(0);

}

//gcc -I ./ main.c lenet.c -lm -o main
int main() {
	
    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (load(lenet, "model.dat")) {
        printf("Error loading model\n");
        free(lenet);
        return 1;
    }

    image input_image;
    FILE *fp = fopen(INPUT_FILE, "rb");
    if (!fp) {
        printf("Error opening input file\n");
        free(lenet);
        return 1;
    }

    // Bỏ qua header của file idx3-ubyte
    fseek(fp, 16, SEEK_SET);
    fread(&input_image, sizeof(image), 1, fp);
    fclose(fp);

    // Dự đoán
    int prediction = Predict(lenet, input_image, 10);
    printf("Predicted digit: %d\n", prediction);

    free(lenet);
    return 0;
}