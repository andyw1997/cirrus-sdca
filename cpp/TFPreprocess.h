//
// Created by andyw1997 on 12/3/18.
//

#ifndef CPP_TFPREPROCESS_H
#define CPP_TFPREPROCESS_H
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <sstream>
#include "cirrus/src/Configuration.h"
#include "cirrus/src/SparseDataset.h"

class TFPreprocess {
public:
    cirrus::SparseDataset read_criteo_sparse_tf(const std::string& input_file,
            const std::string& delimiter,
            const cirrus::Configuration& config);
private:
    void parse_criteo_tf_line(
            const std::string& line, const std::string& delimiter,
            std::vector<std::pair<int, int64_t>>& output_features,
            uint32_t& label, const cirrus::Configuration& config);
    void read_criteo_tf_thread(std::ifstream& fin, std::mutex& fin_lock,
            const std::string& delimiter,
            std::vector<std::vector<std::pair<int, int64_t>>>& samples_res,
            std::vector<uint32_t>& labels_res,
            uint64_t limit_lines, std::atomic<unsigned int>& lines_count,
            std::function<void(const std::string&, const std::string&,
                    std::vector<std::pair<int, int64_t>>&, uint32_t&)> fun);
    void preprocess(std::vector<std::vector<std::pair<int, int64_t>>>& samples);
    int find_bucket(int64_t value, const std::vector<float>& buckets);
    void print_sparse_sample(std::vector<std::pair<int, int64_t>> sample);

    template<class C>
    C hex_string_to(const char* s) {
        std::stringstream ss;
        ss << std::hex << s;

        C c;
        ss >> c;
        return c;
    }
};


#endif //CPP_TFPREPROCESS_H
