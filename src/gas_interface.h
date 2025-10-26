#pragma once
#include "graph.h"
namespace gaslib {

template<typename dist_t> 
class GASStorage;

template<typename dist_t, typename filter_t> 
requires GASFilterConcept<filter_t>
struct GASTree;

template<typename dist_t> 
class StatusList;

}