//
// Created by andyw1997 on 11/28/18.
//

#include <random>
#include <algorithm>
#include <cmath>
#include "SDCA.h"

SDCA::SDCA() {}

void SDCA::train(const SparseMatrix &data, std::vector<float> &a_0, int epochs, float lamb) {
    SparseMatrix x (data);
    std::vector<float> y = data.get_labels();
    for (int i = 0; i < y.size(); i++) {
        if (y[i] == float(0.0)) {
            y[i] = (float) -1.0;
        }
    }
    compute_updates(x, y, a_0, epochs, lamb);
}

float SDCA::log_loss(const SparseMatrix &data) {
    std::vector<float> y = data.get_labels();
    std::vector<float> p_vals = get_p_vals(data);
    float total = 0;

    for (int i = 0; i < p_vals.size(); i++) {
        float p = std::max(std::min(p_vals[i], (float) (1. - 10e-12)), (float) 10e-12);
        if (y[i] == 1.) {
            total -= std::log(p);
        } else {
            total -= std::log(1. - p);
        }
    }

    return total / (float) y.size();
}

std::vector<float> SDCA::predict(const SparseMatrix &data) {
    std::vector<float> values = data.dot(w);
    for (int i = 0; i < values.size(); i++) {
        if (values[i] < 0) {
            values[i] = 0.0;
        } else {
            values[i] = 1.0;
        }
    }
    return values;
}

std::vector<float> SDCA::get_p_vals(const SparseMatrix &data) {
    std::vector<float> values = data.dot(w);
    for (int i = 0; i < values.size(); i++) {
        values[i] = sigmoid(values[i]);
    }
    return values;
}

float SDCA::sigmoid(float z) {
    return (float) (1.0/(1.0 + exp(-1 * z)));
}

void SDCA::compute_updates(SparseMatrix &x, std::vector<float> &y, std::vector<float> &a, int epochs, float lamb) {
    auto len_a = (float) a.size();
    float scaling_factor = (float) 1.0 / (lamb * len_a);
    std::vector<float> w = x.multiply(a);
    for (int i = 0; i < w.size(); i++) {
        w[i] *= scaling_factor;
    }

    std::vector<int> permutation((unsigned long) len_a);
    std::iota(permutation.begin(), permutation.end(), 0);

    std::random_device rd;
    std::minstd_rand g(rd());

    for (int k = 0; k < epochs; k++) {
        std::shuffle(permutation.begin(), permutation.end(), g);

        for (int i = 0; i < len_a; i++) {
            compute_update(permutation[i], x, y, a, w, lamb);
        }
    }

    this->a = a;
    this->w = w;
}

void SDCA::compute_update(int index, SparseMatrix &x, std::vector<float> &y, std::vector<float> &a, std::vector<float> &w, float lamb) {
    auto n = (float) a.size();

    float numerator = (1.0f + std::exp(x.dot_with_row(index, w) * y[index]));
    numerator = y[index] / numerator - a[index];
    float denominator = std::max(1.0f, 0.25f + (x.norm_squared(index) / (lamb * n)));

    float a_grad = numerator/denominator;

    a[index] += a_grad;

    std::vector<float> w_update = x.get_scaled_row(index, a_grad / (lamb * n));

    for (int i = 0; i < w.size(); i++) {
        w[i] += w_update[i];
    }
}