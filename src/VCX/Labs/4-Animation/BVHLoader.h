#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace VCX::Labs::Animation {

    enum class BVHChannelType : std::uint8_t {
        Xposition,
        Yposition,
        Zposition,
        Xrotation,
        Yrotation,
        Zrotation,
    };

    struct BVHNode {
        std::string              Name;
        int                      Parent {-1};
        glm::vec3                Offset {0.f};
        std::vector<BVHChannelType> Channels;
        int                      ChannelOffset {-1}; // offset into per-frame channel array
        bool                     IsEndSite {false};
    };

    struct BVHMotion {
        std::vector<BVHNode> Nodes;
        int                  Root {0};
        int                  NumFrames {0};
        float                FrameTime {1.f / 30.f};
        int                  NumChannels {0};
        // flat array: frame-major. size = NumFrames * NumChannels
        std::vector<float>   ChannelsPerFrame;

        float Duration() const { return NumFrames * FrameTime; }
        bool  Valid() const { return !Nodes.empty() && NumFrames > 0 && NumChannels > 0 && ChannelsPerFrame.size() == std::size_t(NumFrames * NumChannels); }
    };

    struct BVHSampleOptions {
        // scale applied to OFFSETS and POSITION channels (common BVH unit is centimeters)
        float Scale {0.01f};
        // if true, zero out root X/Z translation (in-place playback)
        bool  InPlace {false};
        // rotate the whole pose by -90 degrees around X (useful for some Z-up BVH files)
        bool  RotateZUpToYUp {false};
    };

    struct BVHSampleResult {
        std::vector<glm::vec3> JointPositions; // size = Nodes.size()
        // global transforms are optional for callers that need them
        std::vector<glm::mat4> GlobalTransforms;
    };

    // Load a BVH file from disk. Returns std::nullopt if parsing fails.
    std::optional<BVHMotion> LoadBVH(std::filesystem::path const & filePath, std::string * error = nullptr);

    // Sample pose at time t (seconds). t can be any value; loop handling is up to caller.
    // This function performs simple linear interpolation between frames.
    BVHSampleResult SampleBVH(BVHMotion const & motion, float tSeconds, BVHSampleOptions const & opt);

} // namespace VCX::Labs::Animation
