#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include "gas_interface.h"

namespace gaslib {

/**
 * @brief Output utilities for consistent benchmark reporting
 */
class OutputUtils {
public:
    static constexpr size_t WIDTH = 22;  // Table column width
    static constexpr int NAME_WIDTH = 20;
    static constexpr int EF_WIDTH = 10;
    static constexpr int VALUE_WIDTH = 9;
    static constexpr int UNIT_WIDTH = 10;
    static constexpr int BIG_UNIT_WIDTH = 15;

    /**
     * @brief Print table header for ef parameters
     * @param efs Vector of ef values
     */
    template<typename T>
    static void print_ef_header(const std::vector<T> &efs);

    /**
     * @brief Print separator line between different ef sections
     */
    static void print_ef_separator();

    /**
     * @brief Format recall and latency for consistent output
     * @param recall Recall value [0,1]
     * @param time_ns Time in nanoseconds
     * @return Formatted string (recall, time)
     */
    static std::string format_recall_latency(double recall, size_t time_ns);

    /**
     * @brief Print post-search times table
     * @param indices Vector of indices
     * @param efs Vector of ef values
     * @param post_search_times Matrix of post-search times [index][ef]
     * @param should_break Function to determine if should break early
     */
    template<typename DatasetT, typename FilterT, typename BreakFunc>
    static void print_post_search_times(
        const std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> &indices,
        const std::vector<size_t> &efs,
        const std::vector<std::vector<size_t>> &post_search_times,
        BreakFunc should_break
    );

    /**
     * @brief Print search times and recall table
     * @param indices Vector of indices
     * @param efs Vector of ef values
     * @param search_times Matrix of search times [index][ef]
     * @param hits Matrix of hit counts [index][ef]
     * @param total_queries Total number of queries
     * @param k Number of nearest neighbors
     * @param should_break Function to determine if should break early
     */
    template<typename DatasetT, typename FilterT, typename BreakFunc>
    static void print_search_results(
        const std::vector<std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>>> &indices,
        const std::vector<size_t> &efs,
        const std::vector<std::vector<size_t>> &search_times,
        const std::vector<std::vector<size_t>> &hits,
        size_t total_queries,
        size_t k,
        BreakFunc should_break
    );

    /**
     * @brief Print detailed metrics for a specific index and ef
     * @param index Index instance
     * @param ef Ef parameter value
     * @param metrics Vector of metrics from index->renew()
     * @param hit_to_nquery Hit distribution array
     * @param k Number of nearest neighbors
     */
    template<typename DatasetT, typename FilterT>
    static void print_detailed_metrics(
        const std::unique_ptr<gaslib::IIndex<float, DatasetT, FilterT>> &index,
        size_t ef,
        const std::vector<size_t> &metrics,
        const std::vector<size_t> &hit_to_nquery,
        size_t k
    );
};

} // namespace gaslib

// Include implementation
#include "output_utils_impl.h"