#include "Labs/4-Animation/CaseBVH.h"

#include <algorithm>
#include <cstdio>
#include <cctype>
#include <limits>

#include <glm/gtc/type_ptr.hpp>

#include "Engine/app.h"
#include "Labs/Common/ImGuiHelper.h"

namespace VCX::Labs::Animation {

    namespace {
        static bool contains_ci(std::string const & haystack, std::string_view needle) {
            if (needle.empty()) return true;
            auto lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
            std::string n;
            n.reserve(needle.size());
            for (char c : needle) n.push_back(lower(static_cast<unsigned char>(c)));
            // naive search (strings are short)
            for (std::size_t i = 0; i < haystack.size(); i++) {
                std::size_t j = 0;
                while (j < n.size() && i + j < haystack.size()) {
                    if (lower(static_cast<unsigned char>(haystack[i + j])) != n[j]) break;
                    j++;
                }
                if (j == n.size()) return true;
            }
            return false;
        }

        std::vector<std::uint32_t> buildGridIndices(std::size_t lineCount) {
            std::vector<std::uint32_t> idx;
            idx.reserve(lineCount * 2);
            for (std::uint32_t i = 0; i < lineCount * 2; i++) idx.push_back(i);
            return idx;
        }

        void buildGrid(std::vector<glm::vec3> & verts, int halfExtent, float spacing) {
            verts.clear();
            for (int i = -halfExtent; i <= halfExtent; i++) {
                float x = i * spacing;
                // lines parallel to Z
                verts.push_back(glm::vec3(x, 0.f, -halfExtent * spacing));
                verts.push_back(glm::vec3(x, 0.f,  halfExtent * spacing));
                // lines parallel to X
                float z = i * spacing;
                verts.push_back(glm::vec3(-halfExtent * spacing, 0.f, z));
                verts.push_back(glm::vec3( halfExtent * spacing, 0.f, z));
            }
        }
    }

    CaseBVH::CaseBVH() :
        _program(Engine::GL::UniqueProgram({
            Engine::GL::SharedShader("assets/shaders/flat.vert"),
            Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _bonesItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines),
        _jointsItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Points),
        _endSitesItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Points),
        _majorEndSitesItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Points),
        _gridItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines) {

        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

        // init path buffer
        std::snprintf(_filePathBuf.data(), _filePathBuf.size(), "%s", _filePath.c_str());

        // Build a simple ground grid (static geometry, but we still stream for simplicity)
        std::vector<glm::vec3> gridVerts;
        buildGrid(gridVerts, 10, 0.25f);
        _gridItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(gridVerts));
        _gridItem.UpdateElementBuffer(buildGridIndices(gridVerts.size() / 2));

        loadCurrentBVH();
    }

    void CaseBVH::loadCurrentBVH() {
        _loadError.clear();
        _motion.reset();
        _time = 0.f;
        _framePreview = 0;

        std::string err;
        auto m = LoadBVH(_filePath, &err);
        if (!m) {
            _loadError = err.empty() ? "Failed to load BVH." : err;
            return;
        }
        _motion = std::move(*m);
        rebuildRenderItems();
        resetCameraToMotion();
    }

    void CaseBVH::rebuildRenderItems() {
        if (!_motion.has_value()) return;
        auto const & m = *_motion;
        _posePositions.assign(m.Nodes.size(), glm::vec3(0.f));

        // Build element indices for bones: parent-child edges.
        std::vector<std::uint32_t> boneIdx;
        boneIdx.reserve((m.Nodes.size() - 1) * 2);
        for (std::size_t i = 0; i < m.Nodes.size(); i++) {
            int p = m.Nodes[i].Parent;
            if (p >= 0) {
                boneIdx.push_back(std::uint32_t(p));
                boneIdx.push_back(std::uint32_t(i));
            }
        }
        _bonesItem.UpdateElementBuffer(boneIdx);

        // Split "real" joints and End Sites. Additionally highlight some common end-effectors
        // (head / hands / feet) by name so the final effect is closer to the three.js demo.
        std::vector<std::uint32_t> jointIdx;
        std::vector<std::uint32_t> endIdx;
        std::vector<std::uint32_t> majorEndIdx;
        jointIdx.reserve(m.Nodes.size());
        endIdx.reserve(m.Nodes.size());
        majorEndIdx.reserve(m.Nodes.size());
        for (std::size_t i = 0; i < m.Nodes.size(); i++) {
            if (m.Nodes[i].IsEndSite) {
                bool major = false;
                int const p = m.Nodes[i].Parent;
                if (p >= 0) {
                    auto const & pn = m.Nodes[p].Name;
                    major =
                        contains_ci(pn, "head") ||
                        contains_ci(pn, "hand") ||
                        contains_ci(pn, "wrist") ||
                        contains_ci(pn, "foot") ||
                        contains_ci(pn, "ankle") ||
                        contains_ci(pn, "toe");
                }
                (major ? majorEndIdx : endIdx).push_back(std::uint32_t(i));
            } else {
                jointIdx.push_back(std::uint32_t(i));
            }
        }
        _jointsItem.UpdateElementBuffer(jointIdx);
        _endSitesItem.UpdateElementBuffer(endIdx);
        _majorEndSitesItem.UpdateElementBuffer(majorEndIdx);

        // Initial pose
        auto pose = SampleBVH(m, 0.f, _sampleOpt);
        _posePositions = std::move(pose.JointPositions);
        _bonesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
        _jointsItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
        _endSitesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
        _majorEndSitesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
    }

    void CaseBVH::resetCameraToMotion() {
        if (!_motion.has_value()) return;
        // Compute AABB in rest pose (t=0) to choose a reasonable camera distance.
        auto pose = SampleBVH(*_motion, 0.f, _sampleOpt);
        if (pose.JointPositions.empty()) return;
        glm::vec3 mn( std::numeric_limits<float>::infinity());
        glm::vec3 mx(-std::numeric_limits<float>::infinity());
        for (auto const & p : pose.JointPositions) {
            mn = glm::min(mn, p);
            mx = glm::max(mx, p);
        }
        glm::vec3 c = 0.5f * (mn + mx);
        float radius = glm::length(mx - mn) * 0.6f;
        radius = std::max(radius, 0.5f);

        _camera.Target = c + glm::vec3(0, radius * 0.3f, 0);
        _camera.Eye    = _camera.Target + glm::vec3(-radius * 2.2f, radius * 1.6f, radius * 2.2f);
        _cameraManager.Save(_camera);
    }

    void CaseBVH::OnSetupPropsUI() {
        ImGui::Text("BVH Player");
        ImGui::Separator();

        if (ImGui::CollapsingHeader("File", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Quick presets: list motions under assets/motions (the same set is used in Case 2).
            struct Preset {
                char const * Label;
                char const * Path; // nullptr => custom
            };
            static Preset const kPresets[] = {
                {"T-pose",                 "assets/motions/T-pose.bvh"},
                {"Pirouette",              "assets/motions/pirouette.bvh"},
                {"Jogging (0005)",         "assets/motions/0005_Jogging001.bvh"},
                {"Walking (0005)",         "assets/motions/0005_Walking001.bvh"},
                {"SlowTrot (0005)",        "assets/motions/0005_SlowTrot001.bvh"},
                {"BackwardsWalk (0005)",   "assets/motions/0005_BackwardsWalk001.bvh"},
                {"Jog",                    "assets/motions/jog.bvh"},
                {"Running",                "assets/motions/running.bvh"},
                {"Walking",                "assets/motions/walking.bvh"},
                {"Start Walking",          "assets/motions/start walking.bvh"},
                {"Jump",                   "assets/motions/jump.bvh"},
                {"Custom path",            nullptr},
            };

            static int s_lastPreset = -1;
            if (ImGui::Combo("Preset", &_preset,
                             [](void * data, int idx, char const ** out_text) {
                                 auto const * p = reinterpret_cast<Preset const *>(data);
                                 *out_text = p[idx].Label;
                                 return true;
                             },
                             (void *)kPresets, int(sizeof(kPresets) / sizeof(kPresets[0])))) {
                s_lastPreset = _preset;
                if (kPresets[_preset].Path) {
                    _filePath = kPresets[_preset].Path;
                    std::snprintf(_filePathBuf.data(), _filePathBuf.size(), "%s", _filePath.c_str());
                    loadCurrentBVH();
                    resetCameraToMotion();
                }
            }

            ImGui::TextDisabled("Path is relative to the executable working directory.");
            ImGui::InputText("BVH Path", _filePathBuf.data(), _filePathBuf.size());

            if (ImGui::Button("Load / Reload")) {
                _filePath = std::string(_filePathBuf.data());
                loadCurrentBVH();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Camera")) resetCameraToMotion();

            if (!_loadError.empty()) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "Error: %s", _loadError.c_str());
            }

            if (_motion.has_value()) {
                auto const & m = *_motion;
                ImGui::Spacing();
                ImGui::Text("Nodes: %d", int(m.Nodes.size()));
                ImGui::Text("Channels: %d", m.NumChannels);
                ImGui::Text("Frames: %d", m.NumFrames);
                ImGui::Text("FrameTime: %.6f s (%.1f FPS)", m.FrameTime, 1.f / m.FrameTime);
                ImGui::Text("Duration: %.2f s", m.Duration());
            }
        }

        if (_motion.has_value() && ImGui::CollapsingHeader("Playback", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto const & m = *_motion;
            if (ImGui::Button(_playing ? "Pause" : "Play")) _playing = !_playing;
            ImGui::SameLine();
            if (ImGui::Button("Restart")) {
                _time = 0.f;
                _framePreview = 0;
            }
            ImGui::Checkbox("Loop", &_loop);
            ImGui::SliderFloat("Speed", &_speed, 0.f, 3.f, "%.2fx");

            int maxFrame = std::max(0, m.NumFrames - 1);
            if (ImGui::SliderInt("Frame", &_framePreview, 0, maxFrame)) {
                _time = _framePreview * m.FrameTime;
            }

            float timeSec = _time;
            ImGui::SliderFloat("Time (s)", &timeSec, 0.f, m.Duration(), "%.3f");
            if (timeSec != _time) {
                _time = timeSec;
                _framePreview = int(_time / m.FrameTime);
            }
        }

        if (ImGui::CollapsingHeader("Retarget / Coordinate", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Scale", &_sampleOpt.Scale, 0.001f, 0.1f, "%.4f");
            ImGui::Checkbox("In Place (zero root XZ)", &_sampleOpt.InPlace);
            ImGui::Checkbox("Z-up -> Y-up (-90deg X)", &_sampleOpt.RotateZUpToYUp);
            if (ImGui::Button("Apply Options")) {
                rebuildRenderItems();
                resetCameraToMotion();
            }
        }

        if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Show Grid", &_showGrid);
            ImGui::Checkbox("Show Bones", &_showBones);
            ImGui::Checkbox("Show Joints", &_showJoints);
            ImGui::Checkbox("Show End Sites (hands/feet/head tips)", &_showEndSites);
            ImGui::SliderFloat("Joint Size", &_jointSize, 1.f, 10.f);
            ImGui::SliderFloat("End Site Size", &_endSiteSize, 1.f, 14.f);
            ImGui::SliderFloat("Major Tip Size (head/hands/feet)", &_majorEndSiteSize, 1.f, 18.f);
            ImGui::SliderFloat("Bone Width", &_boneWidth, 0.5f, 4.f);
            ImGui::ColorEdit3("Joint Color", glm::value_ptr(_jointColor));
            ImGui::ColorEdit3("End Site Color", glm::value_ptr(_endSiteColor));
            ImGui::ColorEdit3("Major Tip Color", glm::value_ptr(_majorEndSiteColor));
            ImGui::ColorEdit3("Bone Color", glm::value_ptr(_boneColor));
        }
    }

    Common::CaseRenderResult CaseBVH::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        _frame.Resize(desiredSize);

        _cameraManager.Update(_camera);
        _program.GetUniforms().SetByName("u_Projection", _camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _program.GetUniforms().SetByName("u_View", _camera.GetViewMatrix());

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_LINE_SMOOTH);

        if (_motion.has_value()) {
            auto const & m = *_motion;
            float const dt = Engine::GetDeltaTime();
            if (_playing) {
                _time += dt * _speed;
                if (_loop) {
                    float const dur = m.Duration();
                    if (dur > 0.f) {
                        while (_time >= dur) _time -= dur;
                    }
                } else {
                    _time = std::min(_time, m.Duration());
                }
                _framePreview = int(_time / m.FrameTime);
            }

            auto pose = SampleBVH(m, _time, _sampleOpt);
            _posePositions = std::move(pose.JointPositions);
            if (!_posePositions.empty()) {
                _bonesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
                _jointsItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
                _endSitesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
                _majorEndSitesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_posePositions));
            }
        }

        // Draw
        if (_showGrid) {
            glLineWidth(1.f);
            _program.GetUniforms().SetByName("u_Color", _gridColor);
            _gridItem.Draw({ _program.Use() });
        }

        if (_showBones && _motion.has_value()) {
            glLineWidth(_boneWidth);
            _program.GetUniforms().SetByName("u_Color", _boneColor);
            _bonesItem.Draw({ _program.Use() });
        }

        if (_showJoints && _motion.has_value()) {
            glPointSize(_jointSize);
            _program.GetUniforms().SetByName("u_Color", _jointColor);
            _jointsItem.Draw({ _program.Use() });
            glPointSize(1.f);
        }

        if (_showEndSites && _motion.has_value()) {
            // Smaller end sites (fingers, toes, etc.)
            glPointSize(_endSiteSize);
            _program.GetUniforms().SetByName("u_Color", _endSiteColor);
            _endSitesItem.Draw({ _program.Use() });

            // Highlight head/hands/feet tips a bit more.
            glPointSize(_majorEndSiteSize);
            _program.GetUniforms().SetByName("u_Color", _majorEndSiteColor);
            _majorEndSitesItem.Draw({ _program.Use() });

            glPointSize(1.f);
        }

        glLineWidth(1.f);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_DEPTH_TEST);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }

    void CaseBVH::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);
    }

} // namespace VCX::Labs::Animation
