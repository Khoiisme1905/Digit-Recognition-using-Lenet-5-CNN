#include "lenet.h"
#include <stdio.h>
#include <stdlib.h>

#define INPUT_FILE "input.idx3-ubyte"

// Thêm hàm load
int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int main() {
    // Load model
    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (load(lenet, "model.dat")) {
        printf("Error loading model\n");
        return 1;
    }

    // Read input image
    image input_image;
    FILE *fp = fopen(INPUT_FILE, "rb");
    if (!fp) {
        printf("Error opening input file\n");
        return 1;
    }
    
    // Skip header
    fseek(fp, 16, SEEK_SET);
    fread(&input_image, sizeof(image), 1, fp);
    fclose(fp);

    // Predict
    int prediction = Predict(lenet, input_image, 10);
    printf("%d\n", prediction);

    free(lenet);
    return 0;
}