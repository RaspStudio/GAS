#pragma once

namespace gaslib {

template<typename T>
void OutputUtils::print_ef_header(const std::vector<T> &efs) {
    std::cout << std::left << std::setw(WIDTH) << "Index \\ ef";
    for (auto ef : efs) {
        std::ostringstream oss;
        oss << "ef=" << ef;
        std::cout << std::right << std::setw(WIDTH) << oss.str();
    }
    std::cout << "\n";
}

void OutputUtils::print_ef_separator() {
    std::cout << "------------------------------------\n";
}

std::string OutputUtils::format_recall_latency(double recall, size_t time_ns) {
    std::ostringstream oss;
    oss << "("
        << std::fixed << std::setprecision(4) << recall << ", "
        << std::right << std::setw(6) << (time_ns > 100000 ? time_ns / 1000 : time_ns) 
                                      << (time_ns > 100000 ? " us)" : " ns)");
    return oss.str();
}

template<typename DatasetT, typename FilterT, typename BreakFunc>
void OutputUtils::print_post_search_times(
    const std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> &indices,
    const std::vector<size_t> &efs,
    const std::vector<std::vector<size_t>> &post_search_times,
    BreakFunc should_break
) {
    std::cout << "Post-Search Times (ms):\n";
    print_ef_header(efs);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        std::cout << std::left << std::setw(WIDTH) << indices[i]->name();
        for (size_t j = 0; j < efs.size(); ++j) {
            size_t post_search_time = (post_search_times[i][j] > 100) ? post_search_times[i][j] : 0;
            std::string post_search_time_str = (post_search_time > 1000 || post_search_time == 0) ?
                std::to_string((post_search_time + 999) / 1000) + " ms" :
                std::to_string(post_search_time) + " us";
            std::cout << std::right << std::setw(WIDTH) << post_search_time_str;
            if (should_break(i, j)) break;
        }
        std::cout << "\n";
    }
}

template<typename DatasetT, typename FilterT, typename BreakFunc>
void OutputUtils::print_search_results(
    const std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> &indices,
    const std::vector<size_t> &efs,
    const std::vector<std::vector<size_t>> &search_times,
    const std::vector<std::vector<size_t>> &hits,
    size_t total_queries,
    size_t k,
    BreakFunc should_break
) {
    std::cout << "\nSearch Times (ns):\n";
    print_ef_header(efs);

    for (size_t i = 0; i < indices.size(); ++i) {
        std::cout << std::left << std::setw(WIDTH) << indices[i]->name();
        for (size_t j = 0; j < efs.size(); ++j) {
            double recall = static_cast<double>(hits[i][j]) / (total_queries * k);
            size_t time_per_query_ns = search_times[i][j] / total_queries;
            
            std::string formatted = format_recall_latency(recall, time_per_query_ns);
            std::cout << std::right << std::setw(WIDTH) << formatted;

            if (should_break(i, j)) break;
        }
        std::cout << "\n" << std::flush;
    }
}

template<typename DatasetT, typename FilterT>
void OutputUtils::print_detailed_metrics(
    const std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>> &index,
    size_t ef,
    const std::vector<size_t> &metrics,
    const std::vector<size_t> &hit_to_nquery,
    size_t k
) {
    // Print hit distribution
    std::cout << std::left << std::setw(NAME_WIDTH) << index->name();
    std::cout << std::right << std::setw(EF_WIDTH) << ("(ef=" + std::to_string(ef) + ")");

    for (size_t i = 0; i <= k; ++i) {
        if (hit_to_nquery[i] == 0) continue;
        std::cout << std::right << std::setw(VALUE_WIDTH) << hit_to_nquery[i]
                << " " << std::left << std::setw(UNIT_WIDTH) << ("hit " + std::to_string(i) + ",");
    }
    std::cout << "\n" << std::flush;

    if (metrics.empty()) return;

    // Print basic hops and dists
    std::cout << std::left << std::setw(NAME_WIDTH) << "";
    std::cout << std::right << std::setw(EF_WIDTH) << "";

    std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[0]
            << " " << std::left << std::setw(UNIT_WIDTH) << "hops,";
    std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[1]
            << " " << std::left << std::setw(UNIT_WIDTH) << "dists.";

    // Print detailed metrics if available
    if (metrics.size() >= 17 && metrics[13] > 0) {
        std::cout << "\n";
        std::cout << std::left << std::setw(NAME_WIDTH) << "";
        std::cout << std::right << std::setw(EF_WIDTH) << "";

        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[9]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "upper hops,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[10]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "upper dists,";

        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[11]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "apprch hops,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[12]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "apprch dists,";

        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[13]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "gather hops,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[14]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "gather dists,";

        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[15]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "excld hops,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[16]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "excld dists,";

        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[17]
                << " " << std::left << std::setw(BIG_UNIT_WIDTH) << "pq sorts,";
    }

    // Print case metrics if available
    if (metrics.size() > 2 && metrics[3] > 0) {
        std::cout << "\n";
        std::cout << std::left << std::setw(NAME_WIDTH) << "";
        std::cout << std::right << std::setw(EF_WIDTH) << "";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[2]
                << " " << std::left << std::setw(UNIT_WIDTH) << "anc_hit,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[3]
                << " " << std::left << std::setw(UNIT_WIDTH) << "anc_miss,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[4]
                << " " << std::left << std::setw(UNIT_WIDTH) << "case1,";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[5]
                << " " << std::left << std::setw(UNIT_WIDTH) << "case2,";
                
        std::cout << "\n";
        std::cout << std::left << std::setw(NAME_WIDTH) << "";
        std::cout << std::right << std::setw(EF_WIDTH) << "";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[6]
                << " " << std::left << std::setw(UNIT_WIDTH) << "case3.";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[7]
                << " " << std::left << std::setw(UNIT_WIDTH) << "case4.";
        std::cout << std::right << std::setw(VALUE_WIDTH) << metrics[8]
                << " " << std::left << std::setw(UNIT_WIDTH) << "case5.";
    }

    // Print AE and SC metrics if available
    if (metrics.size() >= 18) {
        std::cout << "\n";
        std::cout << std::left << std::setw(NAME_WIDTH) << "";
        std::cout << std::right << std::setw(EF_WIDTH) << "";
        std::cout << std::right << std::setw(VALUE_WIDTH) << *(metrics.rbegin() + 1)
                << " " << std::left << std::setw(UNIT_WIDTH) << "AEs.";
        std::cout << std::right << std::setw(VALUE_WIDTH) << *metrics.rbegin()
                << " " << std::left << std::setw(UNIT_WIDTH) << "SCs.";
    }
    std::cout << "\n" << std::flush;
}

} // namespace gaslib