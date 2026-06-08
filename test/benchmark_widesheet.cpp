#include "benchmark.h"
#include "benchmark_dataset.h"
#include "gaslib.h"
#include "filter.h"
#include "types.h"
#include "output_utils.h"
#include <iomanip>
#include <filesystem>
#include "perf.h"

// #define ENABLE_PERF

#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
perfmini::Group pg = perfmini::make_default_group();
#endif
#endif

using label_t = gaslib::label_t;
using nodeid_t = gaslib::nodeid_t;

// Global constant for batch processing
constexpr size_t batch_size = 16;

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

        std::string cache_path = (cache_dir + "/" + data_id)
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
    const std::string &gt_path) -> std::vector<std::unordered_set<nodeid_t>> {
    
    size_t n_queries = queryset.size();
    std::vector<std::unordered_set<nodeid_t>> gt(n_queries);    
    std::cout << "----------------------------------------\n" << std::flush;
    std::cout << "Generating ground truth for " << n_queries << " queries with k=" << k << "\n" << std::flush;

    std::string cache_path = gt_path;

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
    unsigned int n_seg,
    unsigned int batch_size,
    std::string query_seq_mode,
    std::map<size_t, std::pair<std::string, std::string>>& query_plan
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
                size_t segment_actual_size = 0;

                std::vector<size_t> segment_hit_to_nquery(k + 1, 0);

                // Prepare query sequence (can be sequential or randomized)
                std::vector<int> query_seq(queryset.size());
                std::iota(query_seq.begin(), query_seq.end(), 0);

                if (query_seq_mode == "reverse"){
                    std::reverse(query_seq.begin(), query_seq.end());
                } else if (query_seq_mode == "random"){
                    std::mt19937 g(42);
                    std::shuffle(query_seq.begin(), query_seq.end(), g);
                }
                
                std::mutex mtx_search_time;
                std::mutex mtx_recall;
                std::mutex mtx_search_post_time;
                std::mutex mtx_search;

                size_t actual_batch_size = batch_size;

                
                for (size_t i = 0; i < queryset.size(); i += actual_batch_size) {
                    
                    
                    if(query_plan.find(i) != query_plan.end()) {
                        std::cout<<"bmeta changed to " << query_plan[i].first << std::endl;
                        std::string bmeta_path = query_plan[i].first;
                        std::string qmeta_path = query_plan[i].second;
                        index->replace_meta(bmeta_path);
                        queryset.replace_filters(qmeta_path);
                    }
                    for (size_t j = i + 1; j < i + batch_size; ++j) {
                        if(query_plan.find(j) != query_plan.end()){
                            actual_batch_size = j - i;
                            break;
                        }
                     }
                    size_t current_batch_end = std::min(i + actual_batch_size, queryset.size());
                    
                    size_t batch_b = k;
                    double disable_frac = 0.3;
                    double frac = static_cast<double>(i) / queryset.size();
                    if (frac <= disable_frac) {
                        batch_b = static_cast<size_t>(ef - (ef - k) * (static_cast<double>(i) / (queryset.size() * disable_frac)));
                    } else {
                        batch_b = static_cast<size_t>(k);
                    }

                    struct QueryResult {
                        std::priority_queue<std::pair<float, label_t>> res_queue;
                        long long search_latency;
                    };
                    std::vector<QueryResult> batch_data(current_batch_end - i);

                    // #pragma omp parallel for
                    for (size_t j = i; j < current_batch_end; ++j) {
                        size_t local_idx = j - i;
                        const float* query = queryset.get_vector(query_seq[j]);
                        auto filter = queryset.get_filter(query_seq[j]);

                        auto start_time = std::chrono::high_resolution_clock::now();
                        
                        std::priority_queue<std::pair<float, label_t>> results;
                        results = index->search(query, k, &filter, ef, batch_b);
                        
                        auto end_time = std::chrono::high_resolution_clock::now();
                        
                        batch_data[local_idx].search_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                        batch_data[local_idx].res_queue = std::move(results);
                    }

                    #pragma omp parallel for
                    for (size_t j = i; j < current_batch_end; ++j) {
                        size_t local_idx = j - i;
                        auto& res = batch_data[local_idx];
                        
                        std::unordered_set<nodeid_t> results_set;
                        while (!res.res_queue.empty()) {
                            results_set.insert(gaslib::nodeid_t(res.res_queue.top().second));
                            res.res_queue.pop();
                        }
                        
                        size_t hit_count = 0;
                        for (nodeid_t lbl : results_set) {
                            if (gt[query_seq[j]].count(lbl) > 0) {
                                ++hit_count;
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(mtx_search_time);
                            search_times[index_idx][ef_idx] += res.search_latency;
                            segment_search_time += res.search_latency;
                            segment_actual_size += 1;
                        }
                        {
                            std::lock_guard<std::mutex> lock(mtx_recall);
                            hits[index_idx][ef_idx] += hit_count;
                            hit_to_nquery[hit_count]++; 
                            segment_hits += hit_count;
                            segment_hit_to_nquery[hit_count]++;
                        }
                    }
               
                    for (size_t j = i; j < current_batch_end; ++j) {
                        const float* query = queryset.get_vector(query_seq[j]);
                        auto filter = queryset.get_filter(query_seq[j]);
                        
                        // Post search - status_list transitions handled internally
                        index->after_search_stat(query, k, &filter, ef);
                        auto post_start_time = std::chrono::high_resolution_clock::now();
                        index->after_search(query, k, &filter, ef);
                        auto post_end_time = std::chrono::high_resolution_clock::now();
                        auto post_latency = std::chrono::duration_cast<std::chrono::microseconds>(post_end_time - post_start_time).count();
                        
                        post_search_times[index_idx][ef_idx] += post_latency;
                        segment_post_search_time += post_latency;
                    }
                    
                    #pragma omp parallel for
                    for (size_t j = i; j < current_batch_end; ++j) {
                        index->status_clean();
                    }
                    
                    bool need_segment_output = n_seg > 1 && (current_batch_end) % segment_size < batch_size;
                    if (need_segment_output) {
                        // Get metrics for this segment without clearing shortcuts
                        std::vector<size_t> segment_metrics = index->get_statistics();
                        
                        // Calculate segment recall and time
                        size_t queries_in_segment = segment_actual_size;
                        double segment_recall = static_cast<double>(segment_hits) / (queries_in_segment * k);
                        size_t segment_time_per_query_ns = segment_search_time / queries_in_segment;
                        
                        // Output intermediate results with "----" separator for ef sections
                        gaslib::OutputUtils::print_ef_separator();
                        std::cout << "Segment Result " << (current_batch_end / segment_size) << "/" << n_seg 
                                << " (queries " << (current_batch_end - segment_actual_size + 1) << "-" << current_batch_end 
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
                        segment_actual_size = 0;
                        std::fill(segment_hit_to_nquery.begin(), segment_hit_to_nquery.end(), 0);
                    }
                } // End of outer batch loop
                
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
    const std::string &cache_dir,
    const std::string &data_path,
    const std::string &q_plan_path,
    const std::string &query_path,
    const std::string &gt_path,
    unsigned int only_run_idx,
    unsigned int repeat,
    unsigned int n_seg,
    unsigned int batch_size,
    std::string query_seq_mode
) {
    std::map<size_t, std::pair<std::string, std::string>> query_plan = gaslib::load_query_plan(q_plan_path);
    std::string bmeta_path = query_plan[0].first;
    std::string qmeta_path = query_plan[0].second;
    
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

    std::cout << "Data Path: " << data_path << "\n"
              << "BMeta Path: " << bmeta_path << "\n"
              << "Dataset Elements: " << dataset.size() << "/" << max_elements << "\n"
              << "Query Path: " << query_path << "\n"
              << "QMeta Path: " << qmeta_path << "\n"
              << "GT Path: " << gt_path << "\n"
              << "Queries: " << queryset.size() << "/" << max_queries << "\n"<< std::flush;
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

    consider([&]() { return std::make_unique<gaslib::GasIndex<DatasetT, FilterT, 1>>(dataset); });
    consider([&]() { return std::make_unique<gaslib::GasIndex<DatasetT, FilterT, 2>>(dataset); });
    consider([&]() { return std::make_unique<gaslib::GasIndex<DatasetT, FilterT, 2, true>>(dataset); });


    build_all_indices(dataset, indices, cache_dir, data_id, bmeta_id);

    auto gt = load_ground_truth<DatasetT, QuerysetT, FilterT>(queryset, k, gt_path);

    search_and_evaluate(queryset, indices, k, efs, gt, repeat, n_seg, batch_size, query_seq_mode, query_plan);

    print_memory_usage();

    std::cout << "========================================\n";
}

// template <
//     typename DatasetT,
//     typename FilterT1,
//     typename FilterT2
// >
// requires gaslib::DataSetConcept<DatasetT> &&
//          gaslib::FilterConcept<FilterT1> &&
//          gaslib::FilterConcept<FilterT2>


int main(int argc, char **argv) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " dim max_elements max_queries k cache_dir data_path query_plan_path query_path qmeta_path"
                  << " [bmeta2_path qmeta2_path]"
                  << " [only_run_idx] [repeat] [n_seg] [batch_size] [query_seq_mode] [efs...]\n";
        return EXIT_FAILURE;
    }

    size_t dim              = std::stoul(argv[1]);
    size_t max_elements     = std::stoul(argv[2]);
    size_t max_queries      = std::stoul(argv[3]);
    size_t k                = std::stoul(argv[4]);
    std::string cache_dir   = argv[5];
    std::string data_path   = argv[6];
    std::string q_plan_path = argv[7];
    std::string query_path  = argv[8];
    std::string gt_path     = argv[9];

    bool has_composed_meta = false;

    int argi = 10; 

    unsigned int only_run_idx = (argc > argi ? std::stoul(argv[argi]) : 0xFFFFFFFF);
    unsigned int repeat       = (argc > argi + 1 ? std::stoul(argv[argi + 1]) : 1);
    unsigned int n_seg        = (argc > argi + 2 ? std::stoul(argv[argi + 2]) : 1);
    unsigned int batch_size   = (argc > argi + 3 ? std::stoul(argv[argi + 3]) : 1);
    std::string query_seq_mode = (argc > argi + 4 ? std::string(argv[argi + 4]) : "normal");
    std::vector<size_t> efs;

    if (argc > argi + 5) {
        for (int i = argi + 5; i < argc; ++i) {
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
              << "Composed Meta: " << (has_composed_meta ? "true" : "false") << "\n"
              << "Only Run Index: " << only_run_idx << "\n"
              << "Repeat: " << repeat << "\n"
              << "N_seg: " << n_seg << "\n"
              << "Batch_size: " << batch_size << "\n" 
              << "Query Sequence Mode: " << query_seq_mode << "\n" 
              << "Data Path: " << data_path << "\n"
              << "Query Path: " << query_path << "\n"
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

    std::cout << "Running WideSheet Query Benchmark...\n";
    std::cout << "----------------------------------------\n" << std::flush;
    run_benchmark<
        gaslib::FvecsDatasetWithMeta,
        gaslib::FvecsRangeQueryset,
        gaslib::RangeGasFilterFunctor
    >(dim, max_elements, max_queries, k, efs, cache_dir,
        data_path, q_plan_path, query_path, gt_path,
        only_run_idx, repeat, n_seg, batch_size, query_seq_mode);
    

#ifdef ENABLE_PERF
#ifndef RUN_SCRIPT_BUILD
    pg.read_and_print(stderr);
#endif
#endif
    return EXIT_SUCCESS;
}
