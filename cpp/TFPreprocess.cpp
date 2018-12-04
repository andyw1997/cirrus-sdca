//
// Created by andyw1997 on 12/3/18.
//

#include <iostream>
#include "TFPreprocess.h"
#include <cassert>
#include <atomic>
#include "cirrus/src/SparseDataset.h"
#include "cirrus/src/Configuration.h"
#include "cirrus/src/Utils.h"
#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <algorithm>
#include <map>
#include <iomanip>
#include <functional>
#include <bits/stdc++.h>

static const int MAX_STR_SIZE = 10000;

/** Here we read the criteo kaggle dataset and return a sparse dataset
  * We mimick the preprocessing TF does. Differences:
  * 1. We do one hot encoding of each feature
  * 2. We ignore CATEGORICAL features that don't appear more than 15 times (DONE)
  * 3. We don't do crosses for now
  * 4. We bucketize INTEGER feature values using (from tf code):
  * boundaries = [1.5**j - 0.51 for j in range(40)]
  */

/**
  * How to expand features. For every sample we do
  * integer features are expanded in a one hot encoding fashion
  * categorical features the same
  */
cirrus::SparseDataset TFPreprocess::read_criteo_sparse_tf(const std::string& input_file,
                                                 const std::string& delimiter,
                                                 const cirrus::Configuration& config) {
    std::cout << "Reading input file: " << input_file << std::endl;
    std::cout << "Limit_line: " << config.get_limit_samples() << std::endl;

    // we enforce knowing how many lines we read beforehand
    //if (config.get_limit_samples() != 45840618) {
    //  throw std::runtime_error("Wrong number of lines");
    //}

    std::ifstream fin(input_file, std::ifstream::in);
    if (!fin) {
        throw std::runtime_error("Error opening input file");
    }

    std::mutex fin_lock;
    std::atomic<unsigned int> lines_count(0); // count lines processed
    std::vector<std::vector<std::pair<int, int64_t>>> samples;  // final result
    std::vector<uint32_t> labels;                               // final result

    uint64_t num_lines = config.get_limit_samples();

    /* We first read the whole dataset to memory
     * to do the hot encondings etc..
     */

    // create multiple threads to process input file
    std::vector<std::shared_ptr<std::thread>> threads;
    uint64_t nthreads = 20;
    for (uint64_t i = 0; i < nthreads; ++i) {
        threads.push_back(
                std::make_shared<std::thread>(
                        std::bind(&TFPreprocess::read_criteo_tf_thread, this,
                                  std::placeholders::_1, std::placeholders::_2,
                                  std::placeholders::_3, std::placeholders::_4,
                                  std::placeholders::_5, std::placeholders::_6,
                                  std::placeholders::_7, std::placeholders::_8),
                        std::ref(fin), std::ref(fin_lock),
                        std::ref(delimiter), std::ref(samples),
                        std::ref(labels), config.get_limit_samples(), std::ref(lines_count),
                        std::bind(&TFPreprocess::parse_criteo_tf_line, this,
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3,
                                  std::placeholders::_4,
                                  config)
                ));
    }

    for (auto& t : threads) {
        t->join();
    }

    std::cout << "Printing 10 samples before preprocessing" << std::endl;
    for (int i = 0; i < 10; ++i) {
        print_sparse_sample(samples[i]);
        std::cout << std::endl;
    }

    std::cout << "Preprocessing dataset" << std::endl;
    preprocess(samples);
    std::cout << "Preprocessed done" << std::endl;

    std::cout << "Printing 10 samples after preprocessing" << std::endl;
    for (int i = 0; i < 10; ++i) {
        print_sparse_sample(samples[i]);
        std::cout << std::endl;
    }


    std::cout << "Transforming to float.." << std::endl;
    /**
      * FIX THIS
      */
    std::vector<std::vector<std::pair<int, FEATURE_TYPE>>> samples_float;
    std::vector<FEATURE_TYPE> labels_float;
    for (int i = samples.size() - 1; i >= 0; --i) {
        // new sample
        std::vector<std::pair<int, FEATURE_TYPE>> new_vec;
        new_vec.reserve(samples[i].size());
        if (samples[i].size() == 0) {
            throw std::runtime_error("empty sample");
        }

        for (const auto& v : samples[i]) {
            new_vec.push_back(
                    std::make_pair(
                            v.first,
                            static_cast<FEATURE_TYPE>(v.second)));
        }

        // last sample becomes first and so on
        samples_float.push_back(new_vec);
        samples.pop_back();

        labels_float.push_back(labels[i]);
        labels.pop_back();
    }
    std::cout << "Returning.." << std::endl;
    std::cout << "samples_float size: " << samples_float.size() << std::endl;
    std::cout << "labels_float size: " << labels_float.size() << std::endl;

    cirrus::SparseDataset ret(std::move(samples_float), std::move(labels_float));
    // we don't normalize here
    return ret;
}

void TFPreprocess::preprocess(
        std::vector<std::vector<std::pair<int, int64_t>>>& samples) {

    std::vector<std::map<int64_t, uint32_t>> col_freq_count;
    col_freq_count.resize(40); // +1 for bias

    // we compute frequencies of values for each column
    for (const auto& sample : samples) {
        for (const auto& feat : sample) {
            int64_t col = feat.first;
            assert(col >= 0);
            int64_t val = feat.second;
            col_freq_count.at(col)[val]++;
        }
    }

    /**
     * We expand each feature left to right
     */

    // we first go sample by sample and bucketize the integer features
    // in the process we expand each integer feature
    // because each integer feature falls into a single bucket the index
    // for that feature
    std::vector<float> buckets = {
            0.0f, 0.49f, 0.99f, 1.74f, 2.865f, 4.5525f, 7.08375f, 10.880625f, 16.5759375f,
            25.11890625f, 37.933359375f, 57.1550390625f, 85.98755859375f, 129.236337890625f,
            194.1095068359375f, 291.41926025390626f, 437.3838903808594f, 656.3308355712891f,
            984.7512533569336f, 1477.3818800354004f, 2216.3278200531004f, 3324.7467300796507f,
            4987.375095119476f, 7481.317642679214f, 11222.231464018821f, 16833.602196028234f,
            25250.65829404235f, 37876.24244106352f, 56814.61866159528f, 85222.18299239293f,
            127833.5294885894f, 191750.54923288408f, 287626.0788493261f, 431439.3732739892f,
            647159.3149109838f, 970739.2273664756f, 1456109.0960497134f, 2184163.8990745707f,
            3276246.103611856f, 4914369.410417783f, 7371554.370626675f};
    std::cout << "buckets size: " << buckets.size() << std::endl;

    /**
      * For every integer feature index:value we find which bucket it belongs to
      * and then we change index to the new bucket index and value to 1
      */
    uint32_t base_index = 0;
    for (int i = 0; i < 13; ++i) {
        for (auto& sample : samples) {
            int& index = sample[i].first;
            int64_t& value = sample[i].second;
            assert(index == i);

            int64_t bucket_id = find_bucket(value, buckets);
            index = base_index + bucket_id;
            //std::cout
            //  << "col: " << i
            //  << " index: " << index
            //  << " bucket_id: " << bucket_id
            //  << " value: " << value
            //  << "\n";
            value = 1;
        }
        base_index += buckets.size() + 1;
    }

    std::cout << "base_index after integer features: " << base_index << std::endl;

    /**
     * Now for each categorical feature we do:
     * 1. ignore if feature doesn't appear at least 15 times
     * 2. integerize
     */
    // first ignore rare features
    for (uint32_t i = 13; i < col_freq_count.size(); ++i) {
        for (std::map<int64_t, uint32_t>::iterator it =
                col_freq_count[i].begin();
             it != col_freq_count[i].end(); ) {
            if (it->second < 15) {
                //std::cout
                //  << "Deleting entry key: " << it->first
                //  << " value: " << it->second
                //  << " col: " << i
                //  << "\n";
                if (it->first < 0) {
                    throw std::runtime_error("Invalid key value");
                }
                it = col_freq_count[i].erase(it);
            } else {
                ++it;
            }
        }
        std::cout << i << ": " << col_freq_count[i].size() << std::endl;
    }

    // then give each value on each column a unique id
    for (uint32_t i = 13; i < col_freq_count.size(); ++i) {
        int feature_id = 0;
        std::map<int64_t, int> col_feature_to_id;
        for (auto& sample : samples) {
            int& feat_key = sample[i].first;
            int64_t& feat_value = sample[i].second;

            // if this value is not int col_freq_count it means it was
            // previously deleted because it didn't appear frequently enough
            if (col_freq_count[i].find(feat_value) == col_freq_count[i].end()) {
                //std::cout << "i : " << i << " value: " << feat_value << " discarded" << "\n";
                feat_key = INT_MIN; // we mark this pair to later remove it
                feat_value = 1;
                continue;
            } else {
                //std::cout << "i : " << i << " value: " << feat_value << " kept" << "\n";
            }

            auto it = col_feature_to_id.find(feat_value);
            if (it == col_feature_to_id.end()) {
                // give new id to unseen feature
                col_feature_to_id[feat_value] = feature_id;
                // update index to this feature in the sample
                // don't forget to add base_id
                feat_key = base_index + feature_id;
                // increment feature_id for next feature
                feature_id++;
            } else {
                feat_key = base_index + it->second;
            }
            feat_value = 1;
        }
        base_index += feature_id; // advance base_id as many times as there were unique_values

        std::cout << "base_index after cat col: " << i
                  << " features: " << base_index << std::endl;
    }

    /**
      * Here we
      * 1: check that all values are 1 due to integerization
      * 2: remove indexes previously marked for removal due to low frequency
      */
    uint32_t count_removed = 0;
    for (auto& sample : samples) {
        for (auto it = sample.begin(); it != sample.end();) {
            assert(it->second == 1);
            if (it->first == INT_MIN) {
                it = sample.erase(it);
                count_removed++;
                if (count_removed % 1000000 == 0) {
                    std::cout << "count_removed: " << count_removed << "\n";
                }
            } else {
                ++it;
            }
        }
        for (auto it = sample.begin(); it != sample.end(); ++it) {
            if (it->first < 0) {
                std::cout << "it->first: " << it->first << std::endl;
                assert(0);
            }
        }
    }
    //for (auto& sample : samples) {
    //    for (auto it = sample.begin(); it != sample.end(); ++it) {
    //        std::cout << it->first << " ";
    //    }
    //    std::cout << "\n";
    //}
}

int TFPreprocess::find_bucket(int64_t value, const std::vector<float>& buckets) {
    //std::cout
    //  << "value: " << value
    //  << " bucket values: "
    //  << buckets[0] << " " << buckets[1] << " " << buckets[2] << " " << buckets[3]
    //  << " buckets size: " << buckets.size()
    //  << "\n";

    int i = 0;
    while (i < buckets.size() && float(value) >= buckets[i]) {
        ++i;
    }
    return i;
}

void TFPreprocess::read_criteo_tf_thread(std::ifstream& fin, std::mutex& fin_lock,
                                        const std::string& delimiter,
                                        std::vector<std::vector<std::pair<int, int64_t>>>& samples_res,
                                        std::vector<uint32_t>& labels_res,
                                        uint64_t limit_lines, std::atomic<unsigned int>& lines_count,
                                        std::function<void(const std::string&, const std::string&,
                                                           std::vector<std::pair<int, int64_t>>&, uint32_t&)> fun) {
    std::vector<std::vector<std::pair<int, int64_t>>> samples;  // final result
    std::vector<uint32_t> labels;                                // final result
    std::string line;
    uint64_t lines_count_thread = 0;
    while (1) {
        fin_lock.lock();
        getline(fin, line);
        fin_lock.unlock();

        // break if we reach end of file
        if (fin.eof())
            break;

        // enforce max number of lines read
        if (lines_count && lines_count >= limit_lines)
            break;

        uint32_t label;
        std::vector<std::pair<int, int64_t>> features;
        fun(line, delimiter, features, label);

        std::ostringstream oss;
        for (const auto& feat : features) {
            oss << feat.first << ":" << feat.second << " ";
        }

        samples.push_back(features);
        labels.push_back(label);

        if (lines_count % 100000 == 0) {
            std::cout << "Read: " << lines_count << "/" << lines_count_thread << " lines." << std::endl;
        }
        ++lines_count;
        lines_count_thread++;
    }

    fin_lock.lock(); // XXX fix this
    for (const auto& l : labels) {
        labels_res.push_back(l);
    }
    for (const auto& s : samples) {
        samples_res.push_back(s);
    }
    fin_lock.unlock();
}

/**
  * We don't do the hashing trick here
  */
void TFPreprocess::parse_criteo_tf_line(
        const std::string& line, const std::string& delimiter,
        std::vector<std::pair<int, int64_t>>& output_features,
        uint32_t& label, const cirrus::Configuration& config) {
    char str[MAX_STR_SIZE];

    if (line.size() > MAX_STR_SIZE) {
        throw std::runtime_error(
                "Criteo input line is too big: " + std::to_string(line.size()) + " " +
                std::to_string(MAX_STR_SIZE));
    }

    strncpy(str, line.c_str(), MAX_STR_SIZE - 1);
    char* s = str;

    uint64_t col = 0;
    while (char* l = strsep(&s, delimiter.c_str())) {
        if (col == 0) { // it's label
            label = cirrus::string_to<int32_t>(l);
            assert(label == 0 || label == 1);
        } else {
            if (l[0] == 0) { // if feature value is missing
                output_features.push_back(std::make_pair(col - 1, -1));
            } else if (col <= 13) {
                output_features.push_back(std::make_pair(col - 1, cirrus::string_to<int64_t>(l)));
            } else {
                //if (col == 40 && strncmp(l, "ff", 2) == 0) {
                //  std::cout << "col: " << col << " l: " << l << std::endl;
                //}
                int64_t hex_value = hex_string_to<int64_t >(l);
                output_features.push_back(std::make_pair(col - 1, hex_value));
            }
        }
        col++;
    }

    if (config.get_use_bias()) { // add bias constant
        output_features.push_back(std::make_pair(col-2, 1));
    }
}

void TFPreprocess::print_sparse_sample(std::vector<std::pair<int, int64_t>> sample) {
    for (const auto& v : sample) {
        std::cout << v.first << ":" << v.second << " ";
    }
}

