#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// We use the index to represent the token vocab[index]
using Pair = uint64_t;
using Index = uint32_t;
struct PreTokenState {
    std::vector<Index> pre_token_index;
    size_t count;
};

Pair MakePair(Index first, Index second) {
    return (static_cast<Pair>(first) << 32) | second;
}

Index GetFirst(Pair pair) { return static_cast<Index>(pair >> 32); }

Index GetSecond(Pair pair) { return static_cast<Index>(pair); }

struct PairCountCompare {
    PairCountCompare(const std::vector<std::string>& vocab) : vocab(vocab) {}

    bool string_greater(const std::string& s1, const std::string& s2) const {
        size_t len = std::min(s1.size(), s2.size());
        for (size_t i = 0; i < len; ++i) {
            if (s1[i] != s2[i]) {
                return static_cast<unsigned char>(s1[i]) >
                       static_cast<unsigned char>(s2[i]);
            }
        }
        return s1.size() > s2.size();
    }

    bool operator()(const std::pair<Pair, size_t>& a,
                    const std::pair<Pair, size_t>& b) const {
        if (a.second != b.second) {
            return a.second > b.second;
        }

        const std::string& a_first = vocab[GetFirst(a.first)];
        const std::string& b_first = vocab[GetFirst(b.first)];

        if (a_first != b_first) {
            return string_greater(a_first, b_first);
        }

        const std::string& a_second = vocab[GetSecond(a.first)];
        const std::string& b_second = vocab[GetSecond(b.first)];

        return string_greater(a_second, b_second);
    }

   private:
    const std::vector<std::string>& vocab;
};

namespace {
using namespace boost::multi_index;
typedef multi_index_container<
    std::pair<Pair, size_t>,
    indexed_by<hashed_unique<member<std::pair<Pair, size_t>,
                                    Pair,
                                    &std::pair<Pair, size_t>::first>>,
               ordered_non_unique<tag<struct by_count>,
                                  identity<std::pair<Pair, size_t>>,
                                  PairCountCompare>>>
    PairCountContainer;  // a container to store (pair, count) with two indices:
                         // 1. hashed index by pair for fast lookup
                         // 2. ordered index by count for getting max count

}  // namespace

class TokenizerTrainerC {
    // The first 256 must be ascii character, not special tokens.
   public:
    TokenizerTrainerC(size_t target_vocab_size)
        : target_vocab_size(target_vocab_size),
          pair_to_count(boost::make_tuple(
              boost::make_tuple(
                  0,  // 1. bucket_count (0 = auto)

                  boost::multi_index::member<std::pair<Pair, size_t>,
                                             Pair,
                                             &std::pair<Pair, size_t>::first>(),
                  boost::hash<Pair>(),

                  std::equal_to<Pair>()),

              boost::make_tuple(
                  boost::multi_index::identity<std::pair<Pair, size_t>>(),
                  PairCountCompare(vocab)))) {
        for (size_t i = 0; i < 256; i++) {
            AddToken(std::string(1, static_cast<char>(i)));
        }
        vocab.reserve(target_vocab_size);
    };
    ~TokenizerTrainerC() = default;

    void LoadData(const std::vector<py::bytes>& special_tokens,
                  py::dict pre_token_count) {
        for (const auto& sp_token : special_tokens) {
            AddToken(sp_token.cast<std::string>());
        }

        pre_token_state.reserve(pre_token_count.size());
        for (const auto& item : pre_token_count) {
            if (!py::isinstance<py::bytes>(item.first)) {
                throw std::runtime_error(
                    "Pre-token must be of type bytes/string.");
            }
            std::string_view pre_token =
                item.first.cast<py::bytes>().cast<std::string_view>();
            size_t count = item.second.cast<size_t>();

            // We assume id of ascii charactar is its value.
            std::vector<Index> pre_token_index;
            pre_token_index.reserve(pre_token.size());
            for (unsigned char c : pre_token) {
                pre_token_index.push_back(static_cast<Index>(c));
            }
            for (size_t i = 0; i + 1 < pre_token_index.size(); i++) {
                Pair pair =
                    MakePair(pre_token_index[i], pre_token_index[i + 1]);
                UpdatePairCount(pair, count);
                pair_to_containing_pre_tokens_index[pair].push_back(
                    static_cast<Index>(pre_token_state.size()));
            }
            pre_token_state.emplace_back(
                PreTokenState{std::move(pre_token_index), count});
        }
    }

    Pair DetermineMergePair() {
        if (pair_to_count.empty()) return 0;
        Pair max_pair = pair_to_count.get<1>().begin()->first;
        pair_to_count.get<1>().erase(pair_to_count.get<1>().begin());
        return max_pair;
    }

    void UpdatePairCount(Pair pair, ssize_t delta) {
        auto it = pair_to_count.find(pair);
        if (it != pair_to_count.end()) {
            pair_to_count.modify(it,
                                 [&](auto& entry) { entry.second += delta; });
        } else {
            pair_to_count.insert(std::make_pair(pair, delta));
        }
    }

    void MergePair(const Pair pair) {
        std::string merged_token =
            vocab[GetFirst(pair)] + vocab[GetSecond(pair)];
        AddToken(merged_token);
        Index new_index = static_cast<Index>(vocab.size() - 1);

        if (pair_to_containing_pre_tokens_index.find(pair) ==
            pair_to_containing_pre_tokens_index.end()) {
            return;
        }

        const std::vector<Index>& affected_pre_token_indexes =
            pair_to_containing_pre_tokens_index[pair];
        for (const Index pre_token : affected_pre_token_indexes) {
            PreTokenState& state = pre_token_state[pre_token];
            size_t read_i = 0;
            size_t write_i = 0;
            size_t pair_count = 0;
            for (; read_i < state.pre_token_index.size(); read_i++) {
                if (read_i == state.pre_token_index.size() - 1) {
                    // Last token, just copy
                    state.pre_token_index[write_i++] =
                        state.pre_token_index[read_i];
                    continue;
                }
                Pair current_pair = MakePair(state.pre_token_index[read_i],
                                             state.pre_token_index[read_i + 1]);
                if (current_pair != pair) {
                    state.pre_token_index[write_i++] =
                        state.pre_token_index[read_i];
                    continue;
                }
                pair_count++;
                if (write_i > 0) {
                    Pair old_left_pair =
                        MakePair(state.pre_token_index[write_i - 1],
                                 state.pre_token_index[read_i]);
                    UpdatePairCount(old_left_pair, -state.count);
                    Pair new_left_pair =
                        MakePair(state.pre_token_index[write_i - 1], new_index);
                    UpdatePairCount(new_left_pair, state.count);
                    pair_to_containing_pre_tokens_index[new_left_pair]
                        .push_back(pre_token);
                }
                if (read_i + 2 < state.pre_token_index.size()) {
                    Pair old_right_pair =
                        MakePair(state.pre_token_index[read_i + 1],
                                 state.pre_token_index[read_i + 2]);
                    UpdatePairCount(old_right_pair, -state.count);
                    Pair new_right_pair =
                        MakePair(new_index, state.pre_token_index[read_i + 2]);
                    UpdatePairCount(new_right_pair, state.count);
                    pair_to_containing_pre_tokens_index[new_right_pair]
                        .push_back(pre_token);
                }
                state.pre_token_index[write_i++] = new_index;
                read_i++;  // Skip next token as it's merged
            }
            if (pair_count > 0) {
                assert(write_i + pair_count == state.pre_token_index.size());
                state.pre_token_index.resize(state.pre_token_index.size() -
                                             pair_count);
            }
        }
        pair_to_count.erase(pair);
        pair_to_containing_pre_tokens_index.erase(pair);
    }

    std::tuple<std::unordered_map<Index, py::bytes>,
               std::vector<std::tuple<py::bytes, py::bytes>>>
    train() {
        size_t loops = target_vocab_size - vocab.size();
        while (loops > 0 && !pair_to_count.empty()) {
            Pair merge_pair = DetermineMergePair();
            merges.push_back(merge_pair);
            MergePair(merge_pair);
            loops--;
        }

        std::unordered_map<Index, py::bytes> index_to_token;
        for (Index i = 0; i < vocab.size(); i++) {
            index_to_token[i] = vocab[i];
        }

        std::vector<std::tuple<py::bytes, py::bytes>> token_merges;
        token_merges.reserve(merges.size());

        for (const Pair& p : merges) {
            token_merges.emplace_back(py::bytes(vocab[GetFirst(p)]),
                                      py::bytes(vocab[GetSecond(p)]));
        }
        return std::make_tuple(index_to_token, token_merges);
    }

   private:
    void AddToken(std::string token) { vocab.push_back(std::move(token)); }

   private:
    std::vector<std::string> vocab;
    std::vector<PreTokenState> pre_token_state;
    PairCountContainer pair_to_count;
    std::unordered_map<Pair, std::vector<Index>>
        pair_to_containing_pre_tokens_index;
    std::vector<Pair> merges;
    size_t target_vocab_size;
};

PYBIND11_MODULE(tokenizer_cpp, m) {
    m.doc() = "BPE Tokenizer Trainer implemented in C++";

    py::class_<TokenizerTrainerC>(m, "TokenizerTrainerC")
        .def(py::init<size_t>(), py::arg("target_vocab_size"))
        .def("LoadData",
             &TokenizerTrainerC::LoadData,
             py::arg("special_tokens"),
             py::arg("pre_token_count"),
             "Load pre-tokenized data and special tokens.")
        .def("train",
             &TokenizerTrainerC::train,
             "Train the tokenizer and return vocabulary and merges.");
}