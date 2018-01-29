/*
 * BK-tree implementation in C++
 * Copyright (C) 2012 Eiichi Sato
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#ifndef _BK_TREE_H_
#define _BK_TREE_H_
 
#include <map>
#include <cmath>
#include <vector>
 
namespace bk {
 
namespace detail {
 
template <typename KeyType, typename MetricType, typename Distance>
class tree_node
{
private:
    typedef tree_node<KeyType, MetricType, Distance> NodeType;
    
private:
    KeyType value;
    std::map<MetricType, NodeType *> *children;
 
public:
    tree_node(const KeyType &key)
        : value(key), children(NULL) { }
 
    ~tree_node() {
        if (children) {
            for (auto iter = children->begin(); iter != children->end(); ++iter)
                delete iter->second;
            delete children;
        }
    }
 
public:
    bool insert(NodeType *node) {
        if (!node)
            return false;
 
        Distance d;
        MetricType distance = d(node->value, this->value);
        if (distance == 0)
            return false; /* value already exists */
 
        if (!children)
            children = new std::map<MetricType, NodeType *>();
 
        auto iterator = children->find(distance);
        if (iterator == children->end()) {
            children->insert(std::make_pair(distance, node));
            return true;
        }
 
        return iterator->second->insert(node);
    }
 
protected:
    bool has_children() const {
        return this->children && this->children->size();
    }
 
protected:
    void _find_within(std::vector<std::pair<KeyType, MetricType>> &result, const KeyType &key, MetricType d) const {
        Distance f;
        MetricType n = f(key, this->value);
        if (n <= d)
            result.push_back(std::make_pair(this->value, n));
 
        if (!this->has_children())
            return;
 
        for (auto iter = children->begin(); iter != children->end(); ++iter) {
            MetricType distance = iter->first;
            if (n - d <= distance && distance <= n + d)
                iter->second->_find_within(result, key, d);
        }
    }
 
public:
    std::vector<std::pair<KeyType, MetricType>> find_within(const KeyType &key, MetricType d) const {
        std::vector<std::pair<KeyType, MetricType>> result;
        _find_within(result, key, d);
        return result;
    }
 
public:
    void dump_tree(int depth = 0) {
        for (int i = 0; i < depth; ++i)
            std::cout << "    ";
        std::cout << this->value << std::endl;
        if (this->has_children())
            for (auto iter = children->begin(); iter != children->end(); ++iter)
                iter->second->dump_tree(depth + 1);
    }
};
 
template <
    typename KeyType,
    typename MetricType
>
struct default_distance {
    MetricType operator()(const KeyType &ki, const KeyType &kj) {
        return sqrt((ki - kj) * (ki - kj));
    }
};

template <
    typename KeyType,
    typename MetricType
>
struct levenshtein_distance {
  MetricType operator()(const KeyType &s1, const KeyType &s2) {
    {
        const std::size_t len1 = s1.size(), len2 = s2.size();
        std::vector<std::vector<MetricType>> d(len1 + 1, std::vector<MetricType>(len2 + 1));
     
        d[0][0] = 0;
        for(int i = 1; i <= len1; ++i) d[i][0] = i;
        for(int i = 1; i <= len2; ++i) d[0][i] = i;
     
        for(int i = 1; i <= len1; ++i)
            for(int j = 1; j <= len2; ++j)
                          // note that std::min({arg1, arg2, arg3}) works only in C++11,
                          // for C++98 use std::min(std::min(arg1, arg2), arg3)
                          d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
        return d[len1][len2];
    }
  }
};

 
} /* namespace detail */
 
template <
    typename KeyType,
    typename MetricType = double,
    typename Distance = detail::levenshtein_distance<KeyType, MetricType>
>
class BKTree
{
private:
    typedef detail::tree_node<KeyType, MetricType, Distance> NodeType;
 
private:
    NodeType *m_top;
    size_t m_n_nodes;
 
public:
    BKTree() : m_top(NULL), m_n_nodes(0) { }
 
public:
    void insert(const KeyType &key) {
        NodeType *node = new NodeType(key);
        if (!m_top) {
            m_top = node;
            m_n_nodes = 1;
            return;
        }
        if (m_top->insert(node))
            ++m_n_nodes;
    };
 
public:
    std::vector<std::pair<KeyType, MetricType>> find_within(KeyType key, MetricType d) const {
        return m_top->find_within(key, d);
    }
 
    void dump_tree() {
        m_top->dump_tree();
    }
 
public:
    size_t size() const {
        return m_n_nodes;
    }
};
 
} /* namespace bk */
 
#endif /* _BK_TREE_H_ */
