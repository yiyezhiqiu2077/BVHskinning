#include "Labs/4-Animation/BVHLoader.h"

#include <charconv>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <sstream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <spdlog/spdlog.h>

#include "Engine/loader.h"

namespace VCX::Labs::Animation {

    namespace {
        // Very small helper tokenizer: BVH is whitespace-separated and uses { }.
        struct TokenStream {
            std::vector<std::string_view> tokens;
            std::size_t                   i {0};

            explicit TokenStream(std::string_view src) {
                // We avoid copying: store string_views into the backing string.
                // Token boundaries: whitespace, and braces are their own tokens.
                std::size_t p = 0;
                while (p < src.size()) {
                    while (p < src.size() && std::isspace(static_cast<unsigned char>(src[p]))) p++;
                    if (p >= src.size()) break;

                    char c = src[p];
                    if (c == '{' || c == '}') {
                        tokens.emplace_back(src.substr(p, 1));
                        p++;
                        continue;
                    }
                    std::size_t q = p;
                    while (q < src.size()) {
                        char d = src[q];
                        if (std::isspace(static_cast<unsigned char>(d)) || d == '{' || d == '}') break;
                        q++;
                    }
                    tokens.emplace_back(src.substr(p, q - p));
                    p = q;
                }
            }

            bool eof() const { return i >= tokens.size(); }

            std::string_view peek() const { return eof() ? std::string_view{} : tokens[i]; }

            std::string_view next() {
                if (eof()) return {};
                return tokens[i++];
            }

            bool consume(std::string_view t) {
                if (peek() == t) {
                    i++;
                    return true;
                }
                return false;
            }
        };

        static std::string_view strip_colon(std::string_view t) {
            if (!t.empty() && t.back() == ':') return t.substr(0, t.size() - 1);
            return t;
        }

        static bool parse_float(std::string_view s, float & out) {
            // std::from_chars for float is C++17 but may be limited on some STL; use stringstream as fallback.
            std::string tmp(s);
            std::stringstream ss(tmp);
            ss >> out;
            return !ss.fail();
        }

        static bool parse_int(std::string_view s, int & out) {
            out = 0;
            auto first = s.data();
            auto last = s.data() + s.size();
            auto [ptr, ec] = std::from_chars(first, last, out);
            return ec == std::errc{} && ptr == last;
        }

        static BVHChannelType parse_channel(std::string_view s) {
            if (s == "Xposition") return BVHChannelType::Xposition;
            if (s == "Yposition") return BVHChannelType::Yposition;
            if (s == "Zposition") return BVHChannelType::Zposition;
            if (s == "Xrotation") return BVHChannelType::Xrotation;
            if (s == "Yrotation") return BVHChannelType::Yrotation;
            return BVHChannelType::Zrotation;
        }

        static glm::vec3 axis_for(BVHChannelType t) {
            switch (t) {
            case BVHChannelType::Xposition:
            case BVHChannelType::Xrotation: return glm::vec3(1, 0, 0);
            case BVHChannelType::Yposition:
            case BVHChannelType::Yrotation: return glm::vec3(0, 1, 0);
            case BVHChannelType::Zposition:
            case BVHChannelType::Zrotation: return glm::vec3(0, 0, 1);
            }
            return glm::vec3(0, 1, 0);
        }

        static bool is_position(BVHChannelType t) {
            return t == BVHChannelType::Xposition || t == BVHChannelType::Yposition || t == BVHChannelType::Zposition;
        }

        static bool is_rotation(BVHChannelType t) {
            return t == BVHChannelType::Xrotation || t == BVHChannelType::Yrotation || t == BVHChannelType::Zrotation;
        }

        static void compute_local_tr(
            BVHMotion const & motion,
            int frameIdx,
            std::vector<glm::vec3> & outT,
            std::vector<glm::quat> & outR,
            BVHSampleOptions const & opt) {
            outT.assign(motion.Nodes.size(), glm::vec3(0.f));
            outR.assign(motion.Nodes.size(), glm::quat(1.f, 0.f, 0.f, 0.f));
            int const base = frameIdx * motion.NumChannels;
            for (std::size_t i = 0; i < motion.Nodes.size(); i++) {
                auto const & n = motion.Nodes[i];
                if (n.ChannelOffset < 0 || n.Channels.empty()) continue;
                glm::vec3 t(0.f);
                glm::quat q(1.f, 0.f, 0.f, 0.f);
                for (std::size_t c = 0; c < n.Channels.size(); c++) {
                    auto const ct = n.Channels[c];
                    float v = motion.ChannelsPerFrame[base + n.ChannelOffset + int(c)];
                    if (is_position(ct)) {
                        if (opt.InPlace && int(i) == motion.Root) {
                            // Keep Y (height), remove X/Z drift.
                            if (ct == BVHChannelType::Xposition || ct == BVHChannelType::Zposition) v = 0.f;
                        }
                        t += axis_for(ct) * (v * opt.Scale);
                    } else if (is_rotation(ct)) {
                        // BVH rotation channels represent *intrinsic (local-axis)* rotations in the
                        // order listed (e.g., Zrotation Xrotation Yrotation). For quaternions, the
                        // most direct implementation is to **post-multiply** the incremental rotation
                        // so each subsequent channel rotates around the joint's *current local axis*.
                        float const rad = glm::radians(v);
                        q = q * glm::angleAxis(rad, axis_for(ct));
                    }
                }
                outT[i] = t;
                outR[i] = q;
            }
        }

        static BVHSampleResult build_global(
            BVHMotion const & motion,
            std::vector<glm::vec3> const & t,
            std::vector<glm::quat> const & r,
            BVHSampleOptions const & opt) {
            BVHSampleResult res;
            res.JointPositions.resize(motion.Nodes.size(), glm::vec3(0.f));
            res.GlobalTransforms.resize(motion.Nodes.size(), glm::mat4(1.f));

            glm::mat4 globalFix(1.f);
            if (opt.RotateZUpToYUp) {
                // Z-up -> Y-up (rotate -90 degrees around X)
                globalFix = glm::rotate(glm::mat4(1.f), glm::radians(-90.f), glm::vec3(1, 0, 0));
            }

            for (std::size_t i = 0; i < motion.Nodes.size(); i++) {
                auto const & n = motion.Nodes[i];
                // IMPORTANT: Some BVH exporters (e.g. Mixamo) include POSITION channels on many joints
                // and bake the joint OFFSET into the per-frame POSITION values. In that case, adding
                // both OFFSET + POSITION would double the bone lengths and push the skeleton far away.
                //
                // Heuristic used here:
                // - If a node has any POSITION channels, we treat its local translation as coming
                //   entirely from the channel data (t[i]) and ignore the static OFFSET.
                // - Otherwise, we use OFFSET (scaled) plus any (rare) extra translation.
                bool hasPos = false;
                for (auto const ct : n.Channels) { if (is_position(ct)) { hasPos = true; break; } }

                glm::vec3 baseOffset = hasPos ? glm::vec3(0.f) : (n.Offset * opt.Scale);

                glm::mat4 local = glm::translate(glm::mat4(1.f), baseOffset);
                local = glm::translate(local, t[i]);
                local = local * glm::mat4_cast(r[i]);

                if (n.Parent >= 0) res.GlobalTransforms[i] = res.GlobalTransforms[n.Parent] * local;
                else res.GlobalTransforms[i] = globalFix * local;

                res.JointPositions[i] = glm::vec3(res.GlobalTransforms[i] * glm::vec4(0, 0, 0, 1));
            }
            return res;
        }
    }

    std::optional<BVHMotion> LoadBVH(std::filesystem::path const & filePath, std::string * error) {
        auto blob = Engine::LoadBytes(filePath);
        if (blob.empty()) {
            if (error) *error = "File not found or empty.";
            return std::nullopt;
        }
        std::string src(reinterpret_cast<char const *>(blob.data()), blob.size());
        TokenStream ts(src);

        BVHMotion motion;
        int channelCursor = 0;
        std::vector<int> stack;

        auto fail = [&](std::string msg) -> std::optional<BVHMotion> {
            if (error) *error = std::move(msg);
            return std::nullopt;
        };

        // Expect HIERARCHY
        if (strip_colon(ts.next()) != "HIERARCHY") {
            return fail("Missing HIERARCHY header.");
        }

        auto parse_node = [&](std::string_view kind) -> std::optional<int> {
            std::string_view name = ts.next();
            if (name.empty()) return std::nullopt;
            BVHNode node;
            node.Name = std::string(name);
            node.Parent = stack.empty() ? -1 : stack.back();
            node.IsEndSite = false;
            int idx = int(motion.Nodes.size());
            motion.Nodes.push_back(std::move(node));
            if (kind == "ROOT") motion.Root = idx;
            return idx;
        };

        // Parse hierarchy.
        while (!ts.eof()) {
            auto tok = strip_colon(ts.next());
            if (tok.empty()) break;
            if (tok == "ROOT" || tok == "JOINT") {
                auto idxOpt = parse_node(tok);
                if (!idxOpt) return fail("Unexpected end while reading node name.");
                int idx = *idxOpt;
                // Expect {
                if (!ts.consume("{")) {
                    // Sometimes there is a newline: tolerate but still require '{'
                    if (ts.next() != "{") return fail("Missing '{' after ROOT/JOINT.");
                }
                stack.push_back(idx);
                continue;
            }
            if (tok == "End") {
                // End Site
                if (strip_colon(ts.next()) != "Site") return fail("Malformed 'End Site'.");
                if (!ts.consume("{")) {
                    if (ts.next() != "{") return fail("Missing '{' after End Site.");
                }
                BVHNode end;
                end.IsEndSite = true;
                end.Parent = stack.empty() ? -1 : stack.back();
                end.Name = motion.Nodes[end.Parent].Name + "_EndSite";
                // Parse until '}'
                while (!ts.eof()) {
                    auto t2 = strip_colon(ts.next());
                    if (t2 == "OFFSET") {
                        float x, y, z;
                        if (!parse_float(ts.next(), x) || !parse_float(ts.next(), y) || !parse_float(ts.next(), z))
                            return fail("Invalid OFFSET in End Site.");
                        end.Offset = glm::vec3(x, y, z);
                    } else if (t2 == "}") {
                        break;
                    }
                }
                motion.Nodes.push_back(std::move(end));
                continue;
            }
            if (tok == "OFFSET") {
                if (stack.empty()) return fail("OFFSET outside of any node.");
                float x, y, z;
                if (!parse_float(ts.next(), x) || !parse_float(ts.next(), y) || !parse_float(ts.next(), z))
                    return fail("Invalid OFFSET values.");
                motion.Nodes[stack.back()].Offset = glm::vec3(x, y, z);
                continue;
            }
            if (tok == "CHANNELS") {
                if (stack.empty()) return fail("CHANNELS outside of any node.");
                int n;
                if (!parse_int(ts.next(), n) || n < 0) return fail("Invalid CHANNELS count.");
                auto & node = motion.Nodes[stack.back()];
                node.ChannelOffset = channelCursor;
                node.Channels.clear();
                for (int i = 0; i < n; i++) {
                    node.Channels.push_back(parse_channel(strip_colon(ts.next())));
                }
                channelCursor += n;
                continue;
            }
            if (tok == "}") {
                if (!stack.empty()) stack.pop_back();
                // When stack is empty and next section starts, we'll break on MOTION.
                continue;
            }
            if (tok == "MOTION") {
                break;
            }
            // Ignore any unknown tokens in hierarchy.
        }

        motion.NumChannels = channelCursor;
        if (motion.Nodes.empty() || motion.NumChannels <= 0) return fail("No nodes/channels parsed.");

        // Parse motion header.
        // Frames: <int>
        std::string_view tFrames = strip_colon(ts.next());
        if (tFrames != "Frames") {
            // Some files might have already consumed MOTION; tolerate if token is Frames.
            if (tFrames != "Frames") return fail("Missing 'Frames:' in MOTION section.");
        }
        int nFrames;
        if (!parse_int(strip_colon(ts.next()), nFrames) || nFrames <= 0) return fail("Invalid frame count.");
        motion.NumFrames = nFrames;

        // Frame Time: <float>
        auto tFrame = strip_colon(ts.next());
        if (tFrame != "Frame") return fail("Missing 'Frame Time:' line.");
        auto tTime = strip_colon(ts.next());
        if (tTime != "Time") {
            // Some files use 'Time:' as one token; allow that.
            if (tTime != "Time") return fail("Missing 'Time:' token.");
        }
        float frameTime;
        if (!parse_float(strip_colon(ts.next()), frameTime) || frameTime <= 0.f) return fail("Invalid frame time.");
        motion.FrameTime = frameTime;

        // Frame data.
        motion.ChannelsPerFrame.resize(std::size_t(motion.NumFrames) * std::size_t(motion.NumChannels));
        for (int f = 0; f < motion.NumFrames; f++) {
            for (int c = 0; c < motion.NumChannels; c++) {
                if (ts.eof()) return fail("Unexpected end of file while reading motion data.");
                float v;
                if (!parse_float(strip_colon(ts.next()), v)) return fail("Invalid motion channel value.");
                motion.ChannelsPerFrame[std::size_t(f) * motion.NumChannels + c] = v;
            }
        }

        spdlog::info("Loaded BVH: {} nodes ({} channels), {} frames, dt={}s", motion.Nodes.size(), motion.NumChannels, motion.NumFrames, motion.FrameTime);
        return motion;
    }

    BVHSampleResult SampleBVH(BVHMotion const & motion, float tSeconds, BVHSampleOptions const & opt) {
        if (!motion.Valid()) return {};
        if (motion.NumFrames == 1) {
            std::vector<glm::vec3> t;
            std::vector<glm::quat> r;
            compute_local_tr(motion, 0, t, r, opt);
            return build_global(motion, t, r, opt);
        }

        float const duration = motion.Duration();
        // Clamp into [0, duration)
        if (duration > 0.f) {
            while (tSeconds < 0.f) tSeconds += duration;
            while (tSeconds >= duration) tSeconds -= duration;
        }

        float const frameF = tSeconds / motion.FrameTime;
        int const f0 = int(std::floor(frameF)) % motion.NumFrames;
        int const f1 = (f0 + 1) % motion.NumFrames;
        float const a = frameF - std::floor(frameF);

        std::vector<glm::vec3> t0, t1;
        std::vector<glm::quat> r0, r1;
        compute_local_tr(motion, f0, t0, r0, opt);
        compute_local_tr(motion, f1, t1, r1, opt);

        std::vector<glm::vec3> t(motion.Nodes.size());
        std::vector<glm::quat> r(motion.Nodes.size());
        for (std::size_t i = 0; i < motion.Nodes.size(); i++) {
            t[i] = glm::mix(t0[i], t1[i], a);
            // Avoid quaternion flipping (slerp chooses the long arc if the two quats are in
            // opposite hemispheres), which can look like jitter.
            glm::quat q1 = r1[i];
            if (glm::dot(r0[i], q1) < 0.f) q1 = -q1;
            r[i] = glm::slerp(r0[i], q1, a);
        }
        return build_global(motion, t, r, opt);
    }

} // namespace VCX::Labs::Animation
