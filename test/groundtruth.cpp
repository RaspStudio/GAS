#include "benchmark.h"
#include "benchmark_dataset.h"
#include "gaslib.h"
#include "filter.h"
#include "types.h"
#include <iomanip>
#include <filesystem>

using label_t = gaslib::label_t;
using nodeid_t = gaslib::nodeid_t;


const std::string CACHE_PATH = "./cache/";



template<typename DatasetT, typename QuerysetT, typename FilterT>
requires gaslib::DataSetConcept<DatasetT> &&
         gaslib::QuerySetConcept<QuerysetT, FilterT> &&
         gaslib::FilterConcept<FilterT>
void generate_ground_truth(
    DatasetT &dataset,
    QuerysetT &queryset,
    size_t k,
    const std::string &cache_dir,
    const std::string &data_id,
    const std::string &bmeta_id,
    const std::string &query_id
) {
    
    size_t n_queries = queryset.size();
    std::vector<std::unordered_set<nodeid_t>> gt(n_queries);    
    std::cout << "----------------------------------------\n" << std::flush;
    std::cout << "Generating ground truth for " << n_queries << " queries with k=" << k << "\n" << std::flush;

    
    std::string cache_path = cache_dir + "gt_" + data_id + "_" + bmeta_id + "_" + query_id + "_k" + std::to_string(k) + ".ivecs";

    
    std::cout << "Generating ground truth, building bruteforce index...\n" << std::flush;
    
    gaslib::GroundTruthIndex<DatasetT, FilterT> gt_idx(dataset);
    gt_idx.build(dataset);
    std::cout << "Brute-force index built for ground truth generation.\n" << std::flush;

    size_t total_time = 0;
    #pragma omp parallel for
    for (size_t qi = 0; qi < n_queries; ++qi) {
        const float *qv = queryset.get_vector(qi);
        auto f = queryset.get_filter(qi);

        std::unordered_set<nodeid_t> cur_gt;
        size_t max_gt = k;
        do {
            max_gt *= 20; 
            if (max_gt > k * 100) {
                std::cerr << "Warning: Too many results for query " << qi << ", max_gt=" << max_gt << "\n" << std::flush;
            }
            
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = gt_idx.search(qv, max_gt, &f, 0); // ef not used
            auto t1 = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            
            std::vector<std::pair<float, nodeid_t>> sorted_res;
            while (!res.empty()) {
                auto result = res.top();
                sorted_res.emplace_back(result.first, result.second);
                res.pop();
            }

            if (sorted_res.size() < k) {
                std::cerr << "Error: Query " << qi << " returned fewer results (" 
                        << sorted_res.size() << ") than requested (" << k << ").\n" << std::flush;
                exit(EXIT_FAILURE);
            }

            std::sort(sorted_res.begin(), sorted_res.end(),
                    [](const auto &a, const auto &b) { return a.first < b.first; });

            if (sorted_res[0].first < std::numeric_limits<float>::epsilon()) {
                std::cerr << "Warning: Query " << qi << " has a result with zero distance with " << sorted_res[0].second
                          << ", possible duplicate vectors in dataset.\n" << std::flush;
                exit(EXIT_FAILURE);
            }

            
            for (size_t j = 0; j < k && j < sorted_res.size(); ++j) {
                cur_gt.insert(gaslib::nodeid_t(sorted_res[j].second));
            }

            for (size_t j = k; j < sorted_res.size(); ++j) {
                if (std::abs(sorted_res[j].first - sorted_res[k - 1].first) < std::numeric_limits<float>::epsilon()) {
                    cur_gt.insert(gaslib::nodeid_t(sorted_res[j].second));
                } else {
                    break; 
                }
            }
        } while (cur_gt.size() == max_gt);

        gt[qi] = std::move(cur_gt);
    }
    std::ofstream out(cache_path, std::ios::binary);
    for (size_t qi = 0; qi < n_queries; ++qi) {
        nodeid_t cnt = gt[qi].size(); if (cnt > k) std::cerr << "Warning: Ground truth size " << cnt << " exceeds k=" << k << " for query " << qi << "\n";
        out.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
        for (nodeid_t lbl: gt[qi]) {
            out.write(reinterpret_cast<const char*>(&lbl), sizeof(lbl));
        }
    }
    std::cout << "Generated and Saved ground truth to cache, avg time: "
                << (total_time / n_queries) << " ns/query\n" << std::flush;
}


int main(int argc, char **argv) {
    
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " dim max_elements max_queries k data_path bmeta_path query_path qmeta_path\n";
        return EXIT_FAILURE;
    }

    
    size_t dim              = std::stoul(argv[1]);
    size_t max_elements     = std::stoul(argv[2]);
    size_t max_queries      = std::stoul(argv[3]);
    size_t k                = std::stoul(argv[4]);
    std::string data_path   = argv[5];
    std::string bmeta_path  = argv[6];
    std::string query_path  = argv[7];
    std::string qmeta_path  = argv[8];


    
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    std::cout << "========================================\n";
    std::cout << "Experiment At " << std::put_time(std::localtime(&t_c), "%Y-%m-%d %H:%M:%S") << "\n"
              << "Dim=" << dim << ", "
              << "K=" << k << "\n"
              << "----------------------------------------\n" << std::flush;
    
    std::string data_norm  = std::filesystem::absolute(data_path).lexically_normal();
    std::string bmeta_norm = std::filesystem::absolute(bmeta_path).lexically_normal();
    std::string query_norm = std::filesystem::absolute(query_path).lexically_normal();
    std::string qmeta_norm = std::filesystem::absolute(qmeta_path).lexically_normal();

    
    std::hash<std::string> hasher;
    std::string data_id = std::to_string(hasher(data_norm + std::to_string(max_elements)));
    std::string bmeta_id = std::to_string(hasher(bmeta_norm + std::to_string(max_elements)));
    std::string query_id = std::to_string(hasher(query_norm + qmeta_norm));

    gaslib::FvecsDatasetWithMeta dataset(
        dim, max_elements, data_path, bmeta_path
    );

    
    std::cout << "Data Path: " << data_norm << "\n"
              << "BMeta Path: " << bmeta_norm << "\n"
              << "Dataset Elements: " << dataset.size() << "/" << max_elements << "\n"
              << "Query Path: " << query_norm << "\n"
              << "QMeta Path: " << qmeta_norm << "\n";

    if (gaslib::qmeta_is_range(qmeta_path)) {
        std::cout << "Running Range Query GT Gen...\n";
        std::cout << "----------------------------------------\n" << std::flush;

        gaslib::FvecsRangeQueryset queryset(
            dim, max_queries, query_path, qmeta_path
        );
        
        std::cout << "Queries: " << queryset.size() << "/" << max_queries << "\n"
                  << "Data ID: " << data_id << "\n"
                  << "BMeta ID: " << bmeta_id << "\n"
                  << "Query ID: " << query_id << "\n";
        std::cout << "----------------------------------------\n" << std::flush;

        generate_ground_truth<gaslib::FvecsDatasetWithMeta, gaslib::FvecsRangeQueryset, gaslib::RangeGASFilterFunctor>(
            dataset, queryset, k,
            CACHE_PATH, data_id, bmeta_id, query_id
        );

    } else {
        std::cout << "Running Tag Query GT Gen...\n";
        std::cout << "----------------------------------------\n" << std::flush;

        gaslib::FvecsTagQueryset queryset(
            dim, max_queries, query_path, qmeta_path
        );

        std::cout << "Queries: " << queryset.size() << "/" << max_queries << "\n"
                  << "Data ID: " << data_id << "\n"
                  << "BMeta ID: " << bmeta_id << "\n"
                  << "Query ID: " << query_id << "\n";
        std::cout << "----------------------------------------\n" << std::flush;
        
        generate_ground_truth<gaslib::FvecsDatasetWithMeta, gaslib::FvecsTagQueryset, gaslib::TagGASFilterFunctor>(
            dataset, queryset, k,
            CACHE_PATH, data_id, bmeta_id, query_id
        );
    }

    return EXIT_SUCCESS;
}
