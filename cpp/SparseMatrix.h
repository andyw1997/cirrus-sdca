//
// Created by andyw1997 on 11/28/18.
//

#ifndef CPP_SPARSEMATRIX_H
#define CPP_SPARSEMATRIX_H
#include "cirrus/src/SparseDataset.h"


class SparseMatrix {
public:
    SparseMatrix(const cirrus::SparseDataset &dataset, uint64_t num_columns);
    std::vector<float> dot(std::vector<float> &vec) const;
    float dot_with_row(int index, std::vector<float> &vec) const;
    std::vector<float> get_scaled_row(int index, float scale) const;
    std::vector<float> get_row(int index) const;
    std::vector<float> multiply(std::vector<float> &vec) const;
    float norm_squared(int index) const;
    std::pair<SparseMatrix, SparseMatrix> get_train_and_test();
    std::vector<float> get_labels() const;
    uint64_t get_num_columns() const;
    uint64_t get_num_rows() const;

private:
    cirrus::SparseDataset dataset;
    uint64_t num_columns;
};


#endif //CPP_SPARSEMATRIX_H
