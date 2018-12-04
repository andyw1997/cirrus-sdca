//
// Created by andyw1997 on 12/3/18.
//

#include "cirrus/src/InputReader.h"
#include "cirrus/src/SparseDataset.h"
#include "cirrus/src/Configuration.h"
#include "SparseMatrix.h"
#include "SDCA.h"
#include "TFPreprocess.h"
#include <cmath>

int main() {
    // Also can be cirrus::InputReader input_reader;
    TFPreprocess input_reader;
    cirrus::Configuration config ("/home/andyw1997/Documents/School/research/cirrus/cirrus-sdca/cpp/cirrus/configs/criteo_kaggle.cfg");
    cirrus::SparseDataset dataset = input_reader.read_criteo_sparse_tf(
            "/home/andyw1997/Documents/School/research/cirrus/cirrus-sdca/criteo/train.txt",
            "\t",
            config);

    SDCA* sdca = new SDCA();
    SparseMatrix* data = new SparseMatrix(dataset, 1ULL << config.get_model_bits());

    std::pair<SparseMatrix, SparseMatrix> train_and_test = data->get_train_and_test();
    delete(data);
    SparseMatrix &train = train_and_test.first;
    SparseMatrix &test = train_and_test.second;

    std::vector<float> a (train.get_num_rows());
    std::fill(a.begin(), a.end(), 0);

    time_t start_time;
    time_t curr_time;
    std::cout << "Starting training" << std::endl;
    std::time(&start_time);
    for (int i = 0; i < 10; i ++) {
        sdca->train(train, a, 1, 0.00001);
        std::cout << "Finished epoch " << i + 1 << std::endl;
        std::time(&curr_time);
        std::cout << "Time since start = " << std::difftime(curr_time, start_time) << std::endl;
        std::cout << "Train loss = " << sdca->log_loss(train) << std::endl;
        std::cout << "Test loss = " << sdca->log_loss(test) << std::endl;

        // Take out time it took to get loss
        time_t temp_time = curr_time;
        std::time(&curr_time);
        start_time += curr_time - temp_time;
    }
}

float log_loss(std::vector<float> predictions, std::vector<float> actual) {
    float total = 0.0;
    for (int i = 0; i < predictions.size(); i++) {
        float p = std::max(std::min(predictions[i], (float)(1. - 10e-12)), (float)10e-12);
        if (actual[i] == 1.) {
            total -= std::log(p);
        } else {
            total -= std::log(1. - p);
        }
    }
    return total / predictions.size();
}
