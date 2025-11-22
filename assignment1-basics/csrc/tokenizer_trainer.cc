
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
// We use the index to represent the token vocab[index]
using Pair = uint64_t;
using Index = uint32_t;

Pair MakePair(Index first, Index second) {
    return (static_cast<Pair>(first) << 32) | second;
}

Index GetFirst(Pair pair) { return static_cast<Index>(pair >> 32); }

Index GetSecond(Pair pair) { return static_cast<Index>(pair); }

struct PreTokenState {
    std::vector<Index> pre_token_index;
    size_t count;
};

class TokenizerTrainerC {
    // The first 256 must be ascii character, not special tokens.
   public:
    TokenizerTrainerC(size_t target_vocab_size)
        : target_vocab_size(target_vocab_size) {
        for (size_t i = 0; i < 256; i++) {
            AddToken(std::string(1, static_cast<char>(i)));
        }
        vocab.reserve(target_vocab_size);
    };
    ~TokenizerTrainerC() = default;

    void LoadData(
        const std::vector<std::string_view>& special_tokens,
        const std::unordered_map<std::string_view, size_t>& pre_token_count) {
        for (const auto& sp_token : special_tokens) {
            AddToken(std::string(sp_token));
        }

        pre_token_state.reserve(pre_token_count.size());
        for (const auto& [pre_token, count] : pre_token_count) {
            // We assume id of ascii charactar is its value.
            std::vector<Index> pre_token_index = {pre_token.begin(),
                                                  pre_token.end()};
            pre_token_state.emplace_back(std::move(pre_token_index), count);
        }
    }

    const Pair& DetermineMergePair() {
        auto max_pair_iter = std::max_element(
            pair_to_count.begin(),
            pair_to_count.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        return max_pair_iter->first;
    }

    void MergePair(const Pair& pair) {
        std::string merged_token = vocab[GetFirst(pair)] + vocab[GetSecond(pair)];
        AddToken(merged_token);

    }

   private:
    void AddToken(std::string token) { vocab.push_back(std::move(token)); }

   private:
    std::vector<PreTokenState> pre_token_state;
    std::unordered_map<Pair, size_t> pair_to_count;
    std::vector<std::string> vocab;
    size_t target_vocab_size;
};
