#ifndef MJX_INTERNAL_TENHOU_WALL_H
#define MJX_INTERNAL_TENHOU_WALL_H

#include <openssl/sha.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mjx::internal {

// Reproduces Tenhou's tile wall from a game's SHUFFLE seed
// ("mt19937ar-sha512-n288-base64,<base64>").
//
// Algorithm (verified bit-exact against Tenhou game logs):
//   1. base64-decode the seed into 624 little-endian uint32; seed mt19937ar
//      via init_by_array.
//   2. For each kyoku (in play order, continuing the MT stream): take 288
//      uint32 `src`; for each of the 9 chunks of 32 uint32 (128 bytes, little
//      endian) compute SHA512 and append the 16 little-endian uint32 of the
//      digest to `rnd` (144 uint32 total).
//   3. Fisher-Yates (low to high) on [0..135]: swap(i, i + rnd[i] % (136 - i)).
//   4. The resulting `yama` reversed is the wall in Mjx's layout, i.e.
//      wall[i] = yama[135 - i] (so wall[52..] are the live draws, wall[130] is
//      the dora indicator, etc.). Dice (cosmetic) are rnd[135], rnd[136].
class TenhouWall {
 public:
  explicit TenhouWall(const std::string& seed_str) {
    const std::string kPrefix = "mt19937ar-sha512-n288-base64,";
    auto comma = seed_str.find(',');
    std::string b64 =
        comma == std::string::npos ? seed_str : seed_str.substr(comma + 1);
    std::vector<std::uint8_t> bytes = Base64Decode(b64);
    std::vector<std::uint32_t> key(bytes.size() / 4);
    for (std::size_t i = 0; i < key.size(); ++i)
      key[i] = std::uint32_t(bytes[4 * i]) |
               (std::uint32_t(bytes[4 * i + 1]) << 8) |
               (std::uint32_t(bytes[4 * i + 2]) << 16) |
               (std::uint32_t(bytes[4 * i + 3]) << 24);
    InitByArray(key);
  }

  // Wall (136 tile ids, Mjx layout) for the n-th kyoku in play order.
  // Must be requested in non-decreasing kyoku order (the MT stream is
  // sequential); results are cached.
  const std::vector<int>& GetWall(int kyoku) {
    while (static_cast<int>(cache_.size()) <= kyoku) GenNext();
    return cache_[kyoku];
  }

 private:
  std::uint32_t mt_[624];
  int mti_ = 625;
  std::vector<std::vector<int>> cache_;

  void InitGenrand(std::uint32_t s) {
    mt_[0] = s;
    for (int i = 1; i < 624; ++i)
      mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) +
               static_cast<std::uint32_t>(i);
    mti_ = 624;
  }

  void InitByArray(const std::vector<std::uint32_t>& key) {
    InitGenrand(19650218u);
    int i = 1, j = 0;
    int k = 624 > static_cast<int>(key.size()) ? 624
                                               : static_cast<int>(key.size());
    for (; k; --k) {
      mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
               key[j] + static_cast<std::uint32_t>(j);
      ++i;
      ++j;
      if (i >= 624) {
        mt_[0] = mt_[623];
        i = 1;
      }
      if (j >= static_cast<int>(key.size())) j = 0;
    }
    for (k = 623; k; --k) {
      mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) -
               static_cast<std::uint32_t>(i);
      ++i;
      if (i >= 624) {
        mt_[0] = mt_[623];
        i = 1;
      }
    }
    mt_[0] = 0x80000000u;
  }

  std::uint32_t Genrand() {
    if (mti_ >= 624) {
      for (int kk = 0; kk < 624; ++kk) {
        std::uint32_t y =
            (mt_[kk] & 0x80000000u) | (mt_[(kk + 1) % 624] & 0x7fffffffu);
        mt_[kk] = mt_[(kk + 397) % 624] ^ (y >> 1);
        if (y & 1u) mt_[kk] ^= 0x9908b0dfu;
      }
      mti_ = 0;
    }
    std::uint32_t y = mt_[mti_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  }

  void GenNext() {
    std::uint8_t src[288 * 4];
    for (int n = 0; n < 288; ++n) {
      std::uint32_t v = Genrand();
      src[4 * n] = v & 0xff;
      src[4 * n + 1] = (v >> 8) & 0xff;
      src[4 * n + 2] = (v >> 16) & 0xff;
      src[4 * n + 3] = (v >> 24) & 0xff;
    }
    std::uint8_t rnd_bytes[9 * 64];
    for (int c = 0; c < 9; ++c)
      SHA512(src + c * 128, 128, rnd_bytes + c * 64);
    std::uint32_t rnd[144];
    for (int i = 0; i < 144; ++i)
      rnd[i] = std::uint32_t(rnd_bytes[4 * i]) |
               (std::uint32_t(rnd_bytes[4 * i + 1]) << 8) |
               (std::uint32_t(rnd_bytes[4 * i + 2]) << 16) |
               (std::uint32_t(rnd_bytes[4 * i + 3]) << 24);
    std::vector<int> yama(136);
    for (int i = 0; i < 136; ++i) yama[i] = i;
    for (int i = 0; i < 135; ++i) {
      int j = i + static_cast<int>(rnd[i] % static_cast<std::uint32_t>(136 - i));
      std::swap(yama[i], yama[j]);
    }
    std::vector<int> wall(136);
    for (int i = 0; i < 136; ++i) wall[i] = yama[135 - i];
    cache_.push_back(std::move(wall));
  }

  static std::vector<std::uint8_t> Base64Decode(const std::string& in) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> rev(256, -1);
    for (int i = 0; i < 64; ++i) rev[static_cast<unsigned char>(chars[i])] = i;
    std::vector<std::uint8_t> out;
    int val = 0, bits = 0;
    for (char c : in) {
      if (c == '=' || c == '\n' || c == '\r') continue;
      int d = rev[static_cast<unsigned char>(c)];
      if (d < 0) continue;
      val = (val << 6) | d;
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        out.push_back(static_cast<std::uint8_t>((val >> bits) & 0xff));
      }
    }
    return out;
  }
};

}  // namespace mjx::internal

#endif  // MJX_INTERNAL_TENHOU_WALL_H
