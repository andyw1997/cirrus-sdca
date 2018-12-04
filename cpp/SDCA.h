//
// Created by andyw1997 on 11/28/18.
//

#ifndef CPP_SDCA_H
#define CPP_SDCA_H

#include <vector>
#include "SparseMatrix.h"

class SDCA {
public:
    std::vector<float> a;
    std::vector<float> w;
    SDCA();
    void train(const SparseMatrix &data, std::vector<float> &a_0, int epochs, float lamb);
    std::vector<float> get_p_vals(const SparseMatrix &data);
    std::vector<float> predict(const SparseMatrix &data);
    float log_loss(const SparseMatrix &data);

private:
    void compute_updates(SparseMatrix &x, std::vector<float> &y, std::vector<float> &a, int epochs, float lamb);
    void compute_update(int index, SparseMatrix &x, std::vector<float> &y, std::vector<float> &a, std::vector<float> &w, float lamb);
    float sigmoid(float z);
};


#endif //CPP_SDCA_H
