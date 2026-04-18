#pragma once

#include <optional>
#include <array>
#include <string>

#include "Engine/Camera.hpp"
#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Labs/4-Animation/BVHLoader.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/OrbitCameraManager.h"

namespace VCX::Labs::Animation {

    class CaseBVH : public Common::ICase {
    public:
        CaseBVH();

        virtual std::string_view const GetName() override { return "BVH Loader & Player"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        void loadCurrentBVH();
        void rebuildRenderItems();
        void resetCameraToMotion();

        // State
        std::array<char, 512> _filePathBuf {};
        // A richer default BVH (close to the three.js demo) is provided in assets.
        std::string _filePath = "assets/motions/pirouette.bvh";
        std::string _loadError;
        std::optional<BVHMotion> _motion;

        int _preset = 0; // UI preset selector

        bool  _playing {true};
        bool  _loop {true};
        float _speed {1.0f};
        float _time {0.f};
        int   _framePreview {0};

        // Display
        bool  _showBones {true};
        bool  _showJoints {true};
        bool  _showEndSites {true};
        float _jointSize {4.f};
        float _endSiteSize {6.f};
        // Highlight "important" end sites (head / hands / feet) a bit more, similar to the
        // three.js skeleton helper demo where tips are clearly visible.
        float _majorEndSiteSize {10.f};
        float _boneWidth {1.f};
        glm::vec3 _jointColor {1.f, 0.65f, 0.2f};
        glm::vec3 _endSiteColor {0.2f, 0.8f, 1.0f};
        glm::vec3 _majorEndSiteColor {0.25f, 1.0f, 0.55f};
        glm::vec3 _boneColor  {0.9f, 0.9f, 0.9f};
        glm::vec3 _gridColor  {0.2f, 0.2f, 0.2f};
        bool  _showGrid {true};

        BVHSampleOptions _sampleOpt;

        // GL resources
        Engine::GL::UniqueProgram     _program;
        Engine::GL::UniqueRenderFrame _frame;
        Engine::Camera                _camera { .Eye = glm::vec3(-2, 1.8f, 2), .Target = glm::vec3(0, 1, 0) };
        Common::OrbitCameraManager    _cameraManager;

        Engine::GL::UniqueIndexedRenderItem _bonesItem;
        Engine::GL::UniqueIndexedRenderItem _jointsItem;
        Engine::GL::UniqueIndexedRenderItem _endSitesItem;
        Engine::GL::UniqueIndexedRenderItem _majorEndSitesItem;
        Engine::GL::UniqueIndexedRenderItem _gridItem;
        std::vector<glm::vec3>              _posePositions;
    };
} // namespace VCX::Labs::Animation
