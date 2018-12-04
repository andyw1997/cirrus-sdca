//
// Created by andyw1997 on 11/28/18.
//

#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(const cirrus::SparseDataset &dataset, uint64_t num_columns) {
    this->dataset = dataset;
    this->num_columns = num_columns;
}

// Takes in a row vector, treats it as a column vector, then returns a column vector as a row vector.
std::vector<float> SparseMatrix::dot(std::vector<float> &vec) const {
    std::vector<float> result (get_num_rows());
    std::fill(result.begin(), result.end(), 0);

    for (uint64_t i = 0; i < get_num_rows(); i++) {
        const std::vector<std::pair<int, float>>& row = dataset.get_row(i);
        for (std::pair<int, float> entry : row) {
            result[i] += entry.second * vec[entry.first];
        }
    }

    return result;
}

// Takes in a row vector, treats it as a column vector, then returns a column vector as a row vector.
float SparseMatrix::dot_with_row(int index, std::vector<float> &vec) const {
    float result = 0;
    const std::vector<std::pair<int, float>>& row = dataset.get_row((uint64_t) index);

    for (std::pair<int, float> entry : row) {
        result += entry.second * vec[entry.first];
    }

    return result;
}

// Returns scaled version of row index
std::vector<float> SparseMatrix::get_row(int index) const {
    const std::vector<std::pair<int, float>>& row = dataset.get_row((uint64_t) index);
    std::vector<float> result(num_columns);
    std::fill(result.begin(), result.end(), 0);

    for (std::pair<int, float> entry : row) {
        result[entry.first] = entry.second;
    }

    return result;
}

// Returns scaled version of row index
std::vector<float> SparseMatrix::get_scaled_row(int index, float scale) const {
    const std::vector<std::pair<int, float>>& row = dataset.get_row((uint64_t) index);
    std::vector<float> result(num_columns);
    std::fill(result.begin(), result.end(), 0);

    for (std::pair<int, float> entry : row) {
        result[entry.first] = entry.second * scale;
    }

    return result;
}

// Multiplies each row by the corresponding index in input vector and sums together
std::vector<float> SparseMatrix::multiply(std::vector<float> &vec) const {
    std::vector<float> result(num_columns);
    std::fill(result.begin(), result.end(), 0);

    for (uint64_t i = 0; i < get_num_rows(); i++) {
        const std::vector<std::pair<int, float>>& row = dataset.get_row(i);
        for (std::pair<int, float> entry : row) {
            result[entry.first] += entry.second * vec[i];
        }
    }

    return result;
}

// Gets sum of squares of row index
float SparseMatrix::norm_squared(int index) const {
    const std::vector<std::pair<int, float>>& row = dataset.get_row((uint64_t) index);
    auto total = (float) 0.0;

    for (std::pair<int, float> entry : row) {
        total += entry.second * entry.second;
    }

    return total;
}

// Splits the matrix into a training and testing matrix, with 1/10 of the values put into the testing, the rest in train
std::pair<SparseMatrix, SparseMatrix> SparseMatrix::get_train_and_test(){
    std::vector<std::vector<std::pair<int, FEATURE_TYPE>>> samples;
    std::vector<FEATURE_TYPE> labels;
    std::vector<std::vector<std::pair<int, FEATURE_TYPE>>> test_samples;
    std::vector<FEATURE_TYPE> test_labels;

    for (uint64_t i = 0; i < get_num_rows(); ++i) {
        if (i % 10 == 0) {
            test_samples.push_back(dataset.data_[i]);
            test_labels.push_back(dataset.labels_[i]);
        }
        else {
            samples.push_back(dataset.data_[i]);
            labels.push_back(dataset.labels_[i]);
        }
    }

    return std::pair<SparseMatrix, SparseMatrix>(
            SparseMatrix(cirrus::SparseDataset(std::move(samples), std::move(labels)), num_columns),
            SparseMatrix(cirrus::SparseDataset(std::move(test_samples), std::move(test_labels)), num_columns));
}

std::vector<float> SparseMatrix::get_labels() const {
    return dataset.labels_;
}

uint64_t SparseMatrix::get_num_columns() const {
    return num_columns;
}

uint64_t SparseMatrix::get_num_rows() const {
    return dataset.num_samples();
}
