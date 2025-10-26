#include "benchmark.h"
#include "benchmark_dataset.h"
#include "gaslib.h"
#include "filter.h"
#include "types.h"
#include "output_utils.h"
#include <iomanip>
#include <filesystem>
// #include "perf.h"

const std::string CACHE_PATH = "./cache/";
// #define ENABLE_PERF

#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
perfmini::Group pg = perfmini::make_default_group(); // TODO: tbd
#endif
#endif

using label_t = gaslib::label_t;
using nodeid_t = gaslib::nodeid_t;


template<typename DatasetT, typename FilterT>
requires gaslib::DataSetConcept<DatasetT> &&
         gaslib::FilterConcept<FilterT>
void build_all_indices(
    DatasetT &dataset,
    std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> &indices,
    const std::string &cache_dir,
    const std::string &data_id,
    const std::string &bmeta_id
) {
    std::cout << "Starting index build phase...\n" << std::flush;
    std::vector<size_t> build_times;
    build_times.resize(indices.size(), 0);

    for (size_t i = 0; i < indices.size(); ++i) {
        auto &idx = indices[i];

        
        auto t0 = std::chrono::high_resolution_clock::now();

        
        std::string cache_path = (cache_dir + data_id)
                                + (idx->supports_meta_change() ? "" : "_" + bmeta_id);
        if (idx->load(cache_path)) {
            
            std::cout << "Loaded index: " << idx->name() << " from cache\n" << std::flush;
            if (idx->supports_meta_change()) {
                idx->replace_meta(dataset);
                std::cout << "Replaced meta for index: " << idx->name() << "\n" << std::flush;
            }
        } else {
            
            std::cout << "Building index: " << idx->name() << "\n" << std::flush;
            idx->build(dataset);
            std::cout << "Index built: " << idx->name() << "\n" << std::flush;
            idx->save(cache_path);
        }        
        
        
        auto t1 = std::chrono::high_resolution_clock::now();
        build_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "Built index: " << idx->name() << " in " << build_times[i] << " ms\n" << std::flush;
    }

    
    std::cout << "----------------------------------------\n" << std::flush;
    std::cout << "Build Times (ms):\n" << std::flush;
    for (size_t i = 0; i < indices.size(); ++i) {
        std::cout << std::left << std::setw(20) << indices[i]->name()
                  << std::right << std::setw(8) << build_times[i] << " ms\n" << std::flush;
    }
}


template<typename DatasetT, typename QuerysetT, typename FilterT>
requires gaslib::DataSetConcept<DatasetT> &&
         gaslib::QuerySetConcept<QuerysetT, FilterT> &&
         gaslib::FilterConcept<FilterT>
auto load_ground_truth(
    QuerysetT &queryset,
    size_t k,
    const std::string &cache_dir,
    const std::string &data_id,
    const std::string &bmeta_id,
    const std::string &query_id
) -> std::vector<std::unordered_set<nodeid_t>> {
    
    
    size_t n_queries = queryset.size();
    std::vector<std::unordered_set<nodeid_t>> gt(n_queries);    
    std::cout << "----------------------------------------\n" << std::flush;
    std::cout << "Generating ground truth for " << n_queries << " queries with k=" << k << "\n" << std::flush;

    
    std::string cache_path = cache_dir + "gt_" + data_id + "_" + bmeta_id + "_" + query_id + "_k" + std::to_string(k) + ".ivecs";

    
    if (std::filesystem::exists(cache_path)) {
        
        std::cout << "Loading ground truth from cache: " << cache_path << "\n" << std::flush;
        std::ifstream in(cache_path, std::ios::binary);
        for (size_t qi = 0; qi < n_queries; ++qi) {
            if (in.eof()) {
                std::cerr << "Warning: Reached end of file before expected for query " << qi << "\n" << std::flush;
                exit(EXIT_FAILURE);
            }
            nodeid_t cnt;
            in.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
            if (cnt < k) {
                std::cerr << "Warning: Ground truth size " << cnt << " too small for query " << qi << "\n" << std::flush;
                exit(EXIT_FAILURE);
            }
            for (size_t j = 0; j < cnt; ++j) {
                nodeid_t lbl = 0;
                in.read(reinterpret_cast<char*>(&lbl), sizeof(lbl));
                gt[qi].insert(lbl);
            }
        }
        std::cout << "Loaded ground truth from cache: " << cache_path << "\n" << std::flush;
    } else {
        throw std::runtime_error("Ground truth cache file not found: " + cache_path);
    }
    return gt;
}


template<typename DatasetT, typename QuerysetT, typename FilterT>
requires gaslib::DataSetConcept<DatasetT> &&
         gaslib::QuerySetConcept<QuerysetT, FilterT> &&
         gaslib::FilterConcept<FilterT>
void search_and_evaluate(
    QuerysetT &queryset,
    std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> &indices,
    size_t k,
    std::vector<size_t> &efs,
    const std::vector<std::unordered_set<nodeid_t>> &gt,
    unsigned int repeat,
    unsigned int n_seg
) {
    for (unsigned rep = 0; rep < repeat; ++rep) {
        std::cout << "========================================\n";
        std::cout << "Repeat " << (rep + 1) << " of " << repeat << "\n";

        
        std::vector<std::vector<size_t>> search_times;
        std::vector<std::vector<size_t>> post_search_times;
        std::vector<std::vector<size_t>> hits;

        search_times.resize(indices.size(), std::vector<size_t>(efs.size(), 0));
        post_search_times.resize(indices.size(), std::vector<size_t>(efs.size(), 0));
        hits.resize(indices.size(), std::vector<size_t>(efs.size(), 0));

        
        auto should_break = [&hits, &queryset, k](size_t index_idx, size_t ef_idx) {
            return ((hits[index_idx][ef_idx] * 10000) / ((queryset.size()) * k)) >= 9999;
        };

        
        for (size_t index_idx = 0; index_idx < indices.size(); ++index_idx) {
            
            auto& index = indices[index_idx];

            
            for (size_t ef_idx = 0; ef_idx < efs.size(); ++ef_idx) {
                
                size_t ef = efs[ef_idx];

                
                std::vector<size_t> hit_to_nquery;
                hit_to_nquery.resize(k + 1, 0); // Initialize hit_to_nquery with zeros
                
                // Calculate segment size for intermediate outputs
                size_t segment_size = queryset.size() / n_seg;
                if (segment_size == 0) segment_size = 1; // Avoid division by zero
                
                // Variables to track segment-specific stats
                size_t segment_hits = 0;
                size_t segment_search_time = 0;
                size_t segment_post_search_time = 0;
                std::vector<size_t> segment_hit_to_nquery(k + 1, 0);
                
                
                for (size_t i = 0; i < queryset.size(); ++i) {
                    const float* query = queryset.get_vector(i);
                    auto filter = queryset.get_filter(i);

                    size_t b = k; // TODO

#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
                    pg.enable();
#endif
#endif

                    auto start_time = std::chrono::high_resolution_clock::now();
                    auto results = index->search(query, k, &filter, ef, b);
                    auto end_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
                    pg.disable();
#endif
#endif
                    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                    search_times[index_idx][ef_idx] += latency;
                    segment_search_time += latency;

                    // Count hits
                    std::unordered_set<nodeid_t> results_set;
                    size_t hit_count = 0;
                    while (!results.empty()) {
                        auto result = results.top();
                        results.pop();
                        results_set.insert(gaslib::nodeid_t(result.second));
                    }
                    for (nodeid_t lbl : results_set) {
                        if (gt[i].count(lbl) > 0) {
                            ++hit_count;
                        }
                    }
                    hits[index_idx][ef_idx] += hit_count;
                    hit_to_nquery[hit_count]++; 
                    segment_hits += hit_count;
                    segment_hit_to_nquery[hit_count]++;
                    
                    // Post search
                    auto post_start_time = std::chrono::high_resolution_clock::now();
                    index->after_search(query, k, &filter, ef);
                    auto post_end_time = std::chrono::high_resolution_clock::now();
                    index->after_search_clean();
                    auto post_latency = std::chrono::duration_cast<std::chrono::microseconds>(post_end_time - post_start_time).count();
                    post_search_times[index_idx][ef_idx] += post_latency;
                    segment_post_search_time += post_latency;
                    
                    // Check if we need to output intermediate results
                    bool need_early_output = i < segment_size && i < queryset.size() * 0.01; // First 0.1% queries always output
                    if (need_early_output) {
                        // Get metrics for this early segment without clearing shortcuts
                        std::vector<size_t> early_metrics = index->get_statistics();
                        
                        // Get recall and time for this early segment
                        double early_recall = static_cast<double>(segment_hits) / ((i + 1) * k);
                        size_t early_time_per_query_ns = segment_search_time / (i + 1);
                        // Output intermediate results with "----" separator for ef sections
                        gaslib::OutputUtils::print_ef_separator();
                        std::cout << "Early Segment Result " << (i + 1) << "/" << queryset.size() 
                                  << " (queries 1-" << (i + 1) << ") for ef=" << ef << "\n";
                        // Print detailed metrics and hit distribution
                        gaslib::OutputUtils::print_detailed_metrics(
                            index, ef, early_metrics, hit_to_nquery, k
                        );
                        // Print recall and time for this early segment in consistent format
                        std::string early_recall_latency_str = gaslib::OutputUtils::format_recall_latency(
                            early_recall, early_time_per_query_ns
                        );
                        std::cout << "Early Segment Summary: " << early_recall_latency_str << " per query\n\n" << std::flush;
                    }

                    bool need_segment_output = n_seg > 1 && (i + 1) % segment_size == 0;
                    if (need_segment_output) {
                        // Get metrics for this segment without clearing shortcuts
                        std::vector<size_t> segment_metrics = index->get_statistics();
                        
                        // Calculate segment recall and time
                        size_t queries_in_segment = segment_size;
                        double segment_recall = static_cast<double>(segment_hits) / (queries_in_segment * k);
                        size_t segment_time_per_query_ns = segment_search_time / queries_in_segment;
                        
                        // Output intermediate results with "----" separator for ef sections
                        gaslib::OutputUtils::print_ef_separator();
                        std::cout << "Segment Result " << ((i + 1) / segment_size) << "/" << n_seg 
                                  << " (queries " << (i + 1 - segment_size + 1) << "-" << (i + 1) 
                                  << ") for ef=" << ef << "\n";

                        // Print detailed metrics and hit distribution
                        gaslib::OutputUtils::print_detailed_metrics(
                            index, ef, segment_metrics, segment_hit_to_nquery, k
                        );

                        // Print recall and time for this segment in consistent format
                        std::string recall_latency_str = gaslib::OutputUtils::format_recall_latency(
                            segment_recall, segment_time_per_query_ns
                        );
                        std::cout << "Segment Summary: " << recall_latency_str << " per query\n\n" << std::flush;
                        
                        // Reset segment tracking variables
                        segment_hits = 0;
                        segment_search_time = 0;
                        segment_post_search_time = 0;
                        std::fill(segment_hit_to_nquery.begin(), segment_hit_to_nquery.end(), 0);
                    }
                }
                
                // Handle final segment if there are remaining queries
                if (n_seg > 1 && queryset.size() % segment_size != 0 && segment_hits > 0) {
                    // Get metrics for the final partial segment without clearing shortcuts
                    std::vector<size_t> final_segment_metrics = index->get_statistics();
                    
                    // Calculate final segment recall and time
                    size_t remaining_queries = queryset.size() % segment_size;
                    double final_segment_recall = static_cast<double>(segment_hits) / (remaining_queries * k);
                    size_t final_segment_time_per_query_ns = segment_search_time / remaining_queries;
                    
                    // Output final segment results with "----" separator
                    gaslib::OutputUtils::print_ef_separator();
                    std::cout << "Final Segment Result " << (queryset.size() / segment_size + 1) << "/" << n_seg 
                              << " (queries " << (queryset.size() - remaining_queries + 1) << "-" << queryset.size() 
                              << ") for ef=" << ef << "\n";
                    
                    // Print detailed metrics and hit distribution 
                    gaslib::OutputUtils::print_detailed_metrics(
                        index, ef, final_segment_metrics, segment_hit_to_nquery, k
                    );

                    // Print recall and time for the final segment in consistent format
                    std::string final_recall_latency_str = gaslib::OutputUtils::format_recall_latency(
                        final_segment_recall, final_segment_time_per_query_ns
                    );
                    std::cout << "Final Segment Summary: " << final_recall_latency_str << " per query\n\n" << std::flush;
                }
                
                // Always call renew() at the end of each ef to clear shortcuts and get final metrics
                std::vector<size_t> metrics = index->renew();
                
                // Output final results for this ef value (only when n_seg == 1)
                if (n_seg == 1) {
                    gaslib::OutputUtils::print_ef_separator();
                    std::cout << "Final Result for ef=" << ef << "\n";
                    
                    // Print detailed metrics and hit distribution
                    gaslib::OutputUtils::print_detailed_metrics(
                        index, ef, metrics, hit_to_nquery, k
                    );
                } // End of n_seg == 1 condition

                if (should_break(index_idx, ef_idx)) break;
            }
        }      
        
        
        std::cout << "========================================\n";
        
        // Print post-search times table using utility function
        gaslib::OutputUtils::print_post_search_times(
            indices, efs, post_search_times, should_break
        );
        
        // Print overall search results table using utility function  
        gaslib::OutputUtils::print_ef_separator();
        gaslib::OutputUtils::print_search_results(
            indices, efs, search_times, hits, queryset.size(), k, should_break
        );
    }
}

void print_memory_usage() {
    std::cout << "----------------------------------------\n";
    std::ifstream file_stream("/proc/self/status");
    std::string line;
    while (std::getline(file_stream, line)) {
        if (line.find("VmHWM") != std::string::npos) {
            size_t begin = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            size_t value = std::stoull(line.substr(begin, end - begin + 1));
            std::cout << "Peak PMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
        }
        if (line.find("VmRSS") != std::string::npos) {
            size_t begin = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            size_t value = std::stoull(line.substr(begin, end - begin + 1));
            std::cout << "Current PMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
        }
        if (line.find("VmPeak") != std::string::npos) {

            size_t begin = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            size_t value = std::stoull(line.substr(begin, end - begin + 1));
            std::cout << "Peak VMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
        }
        if (line.find("VmSize") != std::string::npos) {

            size_t begin = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            size_t value = std::stoull(line.substr(begin, end - begin + 1));
            std::cout << "Current VMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
        }
        if (line.find("VmData") != std::string::npos) {

            size_t begin = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            size_t value = std::stoull(line.substr(begin, end - begin + 1));
            std::cout << "Data Segment VMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
        }
    }
}


template <
    typename DatasetT,
    typename QuerysetT,
    typename FilterT
>
requires gaslib::DataSetConcept<DatasetT> &&
         gaslib::QuerySetConcept<QuerysetT, FilterT> &&
         gaslib::FilterConcept<FilterT>
void run_benchmark(
    size_t dim,
    size_t max_elements,
    size_t max_queries,
    size_t k,
    std::vector<size_t> &efs,
    const std::string &data_path,
    const std::string &bmeta_path,
    const std::string &query_path,
    const std::string &qmeta_path,
    unsigned int only_run_idx,
    unsigned int repeat,
    unsigned int n_seg
) {
    
    DatasetT dataset(dim, max_elements, data_path, bmeta_path);
    QuerysetT queryset(dim, max_queries, query_path, qmeta_path);

    
    std::string data_norm  = std::filesystem::absolute(data_path).lexically_normal();
    std::string bmeta_norm = std::filesystem::absolute(bmeta_path).lexically_normal();
    std::string query_norm = std::filesystem::absolute(query_path).lexically_normal();
    std::string qmeta_norm = std::filesystem::absolute(qmeta_path).lexically_normal();

    
    std::hash<std::string> hasher;
    std::string data_id = std::to_string(hasher(data_norm + std::to_string(max_elements)));
    std::string bmeta_id = std::to_string(hasher(bmeta_norm + std::to_string(max_elements)));
    std::string query_id = std::to_string(hasher(query_norm + qmeta_norm));

    
    std::cout << "Data Path: " << data_norm << "\n"
              << "BMeta Path: " << bmeta_norm << "\n"
              << "Dataset Elements: " << dataset.size() << "/" << max_elements << "\n"
              << "Query Path: " << query_norm << "\n"
              << "QMeta Path: " << qmeta_norm << "\n"
              << "Queries: " << queryset.size() << "/" << max_queries << "\n"
              << "Data ID: " << data_id << "\n"
              << "BMeta ID: " << bmeta_id << "\n"
              << "Query ID: " << query_id << "\n";
    std::cout << "----------------------------------------\n" << std::flush;

    
    std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> indices;

    
    size_t consider_bit = 0;
    auto consider = [&](auto factory) {
        if (only_run_idx & (1u << consider_bit)) {
            auto index = factory();
            std::cout << "Adding index: " << index->name() << "\n" << std::flush;
            indices.push_back(std::move(index));
        }
        ++consider_bit;
    };

    
    consider([&]() { return std::make_unique<gaslib::GASHNSW<DatasetT, FilterT>>(dataset); }); // 2048

    
    build_all_indices(dataset, indices, CACHE_PATH, data_id, bmeta_id);

    
    auto gt = load_ground_truth<DatasetT, QuerysetT, FilterT>(queryset, k, CACHE_PATH, data_id, bmeta_id, query_id);

    
    search_and_evaluate(queryset, indices, k, efs, gt, repeat, n_seg);

    print_memory_usage();

    std::cout << "========================================\n";
}

int main(int argc, char **argv) {
    
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " dim max_elements max_queries k data_path bmeta_path query_path qmeta_path [only_run_idx] [repeat] [n_seg] [efs...]\n";
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
    unsigned int only_run_idx = (argc >= 10 ? std::stoul(argv[9]) : 0xFFFFFFFF);
    unsigned int repeat       = (argc >= 11 ? std::stoul(argv[10]) : 1);
    unsigned int n_seg        = (argc >= 12 ? std::stoul(argv[11]) : 1);

    std::vector<size_t> efs;

    if (argc > 12) {
        
        for (int i = 12; i < argc; ++i) {
            efs.push_back(std::stoul(argv[i]));
        }
    } else {
        
        efs = { 10u*k };
    }

    
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    std::cout << "========================================\n";
    std::cout << "Experiment At " << std::put_time(std::localtime(&t_c), "%Y-%m-%d %H:%M:%S") << "\n"
              << "Dim=" << dim << ", "
              << "K=" << k << "\n"
              << "Only Run Index: " << only_run_idx << "\n"
              << "Repeat: " << repeat << "\n"
              << "N_seg: " << n_seg << "\n"
              << "----------------------------------------\n"
              << "Command: ";
    for (int i = 0; i < argc; ++i) std::cout << argv[i] << " ";
    std::cout << "\n----------------------------------------\n" << std::flush;
              

#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
    if (!pg.open_current_thread()) {
        std::fprintf(stderr, "[perf] open events failed (check /proc/sys/kernel/perf_event_paranoid)\n");
    } else {
        
        fprintf(stderr, "[perf] opened events OK\n");
    }
    pg.reset();
#endif
#endif

    if (gaslib::qmeta_is_range(qmeta_path)) {
        std::cout << "Running Range Query Benchmark...\n";
        std::cout << "----------------------------------------\n" << std::flush;
        run_benchmark<
            gaslib::FvecsDatasetWithMeta,
            gaslib::FvecsRangeQueryset,
            gaslib::RangeGASFilterFunctor
        >(dim, max_elements, max_queries, k, efs,
            data_path, bmeta_path, query_path, qmeta_path,
            only_run_idx, repeat, n_seg);
    } else {
        std::cout << "Running Tag Query Benchmark...\n";
        std::cout << "----------------------------------------\n" << std::flush;
        run_benchmark<
            gaslib::FvecsDatasetWithMeta,
            gaslib::FvecsTagQueryset,
            gaslib::TagGASFilterFunctor
        >(dim, max_elements, max_queries, k, efs,
            data_path, bmeta_path, query_path, qmeta_path,
            only_run_idx, repeat, n_seg);
    }
#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
    pg.read_and_print(stderr);
#endif
#endif
    return EXIT_SUCCESS;
}
