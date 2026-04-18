#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Engine/Camera.hpp"
#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"

#include "Labs/4-Animation/BVHLoader.h"
#include "Labs/4-Animation/GltfSkinnedModel.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/OrbitCameraManager.h"

namespace VCX::Labs::Animation {

    // A new case that retargets BVH motion to a skinned glTF 2.0 character.
    // - GPU skinning for rendering (via assets/shaders/skinned.*)
    // - On-demand CPU skinning for precise geometry queries (raycast/picking)
    class CaseBVHSkinned : public Common::ICase {
    public:
        CaseBVHSkinned();

        virtual std::string_view const GetName() override { return "BVH -> glTF Skinned Character"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        // Loading
        void loadCurrentBVH();
        void loadCurrentGLTF();
        void rebuildRetargetMapping();
        void rebuildRetargetData();
        void resetCameraToContent();

        // Retarget helpers
        void computeBVHBindPose();
        void retargetAtTime(float tSeconds);

        enum class HipMode {
            Root,              // apply BVH translation on skeleton root
            Hip,               // apply BVH translation on hips joint
            SplitXZRoot_YHip,  // XZ on root, Y on hip
        };

        // UI / State
        std::array<char, 512> _bvhPathBuf {};
        std::string           _bvhPath = "assets/motions/jog.bvh";
        std::string           _bvhError;
        std::optional<BVHMotion> _motion;

        std::array<char, 512> _gltfPathBuf {};
        std::string           _gltfPath = "assets/models/character.glb";
        std::string           _gltfError;
        GltfSkinnedModel      _model;

        // Playback
        bool  _playing {true};
        bool  _loop {true};
        float _speed {1.0f};
        float _time {0.f};
        int   _framePreview {0};

        // View
        bool  _showGrid {true};
        bool  _showBVHSkeleton {false};
        bool  _showMesh {true};
        glm::vec4 _meshColor {0.92f, 0.92f, 0.95f, 1.0f};
        glm::vec3 _gridColor  {0.2f, 0.2f, 0.2f};
        glm::vec3 _boneColor  {0.9f, 0.9f, 0.9f};
        float _boneWidth {1.0f};

        // BVH sampling options (same meaning as CaseBVH)
        BVHSampleOptions _sampleOpt;

        // Retarget options
        bool  _useRootMotion {true};
        bool  _inPlaceTarget {false};
        bool  _autoScaleEnabled {true};
        bool  _useDirectionOffsets {false}; // fallback: direction-based bind offsets (can be unstable)
        float _autoScale {1.0f};
        float _userScale {1.0f};
        HipMode _hipMode {HipMode::SplitXZRoot_YHip};
        bool  _preserveHipXZ {false};
        glm::vec3 _bvhToModelEulerDeg {0.0f, 0.0f, 0.0f}; // XYZ degrees

        // Motion preset (UI)
        // Motion preset selector (see UI list in OnSetupPropsUI).
        int _motionPreset {0};

        // Mapping: targetJointIndex (skin order) -> source BVH node index
        std::vector<int> _targetToBVH;

        // BVH bind (local rotations) used for delta computation
        std::vector<glm::quat> _bvhBindLocalRot;
        glm::vec3              _bvhBindRootPos {0.0f};

        // BVH bind (world) after global adjustment
        std::vector<glm::quat> _bvhBindWorldRot;
        std::vector<glm::vec3> _bvhBindWorldPos;

        // Target bind (local/world) used for retarget (optionally with fallback offsets)
        std::vector<glm::quat> _targetBindLocalUsed;
        std::vector<glm::quat> _targetBindWorldRotUsed;
        std::vector<glm::quat> _jointOffsets;

        // Per-joint frame correction: maps target bind-local basis -> BVH bind-local basis
        // Used as: deltaT = inv(C) * deltaS * C
        std::vector<glm::quat> _jointFrameCorr;

        int _bvhHipNode {-1};
        int _targetHipJoint {-1};
        int _targetRootJoint {-1};

        // Cached per-frame target pose (skin order)
        std::vector<glm::quat> _targetLocalRot;
        glm::vec3              _targetRootDelta {0.0f};
        glm::vec3              _targetHipDelta {0.0f};

        // CPU picking / raycast
        bool _enablePicking {false};
        bool _hasHit {false};
        GltfSkinnedModel::RaycastHit _hit;

        // Last framebuffer size (used for correct picking ray construction)
        std::pair<std::uint32_t, std::uint32_t> _lastRenderSize { 1, 1 };

        // GL resources
        Engine::GL::UniqueProgram     _flatProgram;
        Engine::GL::UniqueRenderFrame _frame;
        Engine::Camera                _camera { .Eye = glm::vec3(-2, 1.8f, 2), .Target = glm::vec3(0, 1, 0) };
        Common::OrbitCameraManager    _cameraManager;

        // Debug skeleton overlay (BVH)
        Engine::GL::UniqueIndexedRenderItem _bonesItem;
        Engine::GL::UniqueIndexedRenderItem _gridItem;
        std::vector<glm::vec3>              _bvhPosePositions;
    };
} // namespace VCX::Labs::Animation
