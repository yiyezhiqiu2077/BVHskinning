#include "Labs/4-Animation/CaseBVHSkinned.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Engine/app.h"
#include "Labs/Common/ImGuiHelper.h"

namespace VCX::Labs::Animation {

    namespace {
        static std::vector<std::uint32_t> buildGridIndices(std::size_t lineCount) {
            std::vector<std::uint32_t> idx;
            idx.reserve(lineCount * 2);
            for (std::uint32_t i = 0; i < lineCount * 2; i++) idx.push_back(i);
            return idx;
        }

        static void buildGrid(std::vector<glm::vec3> & verts, int halfExtent, float spacing) {
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

        static glm::quat ExtractRotation(glm::mat4 const & m) {
            glm::mat3 r(m);
            // Orthonormalize columns to be safe
            r[0] = glm::normalize(r[0]);
            r[1] = glm::normalize(r[1]);
            r[2] = glm::normalize(r[2]);
            return glm::normalize(glm::quat_cast(r));
        }

        static glm::vec3 ExtractTranslation(glm::mat4 const & m) {
            return glm::vec3(m[3]);
        }

        static std::string normalizeName(std::string const & in) {
            std::string s;
            s.reserve(in.size());
            for (unsigned char c : in) {
                char lc = static_cast<char>(std::tolower(c));
                if ((lc >= 'a' && lc <= 'z') || (lc >= '0' && lc <= '9')) s.push_back(lc);
            }
            // Strip common rig prefixes
            auto stripPrefix = [&](std::string_view p) {
                if (s.rfind(std::string(p), 0) == 0) s.erase(0, p.size());
            };
            stripPrefix("mixamorig");
            stripPrefix("bip001");
            stripPrefix("armature");
            stripPrefix("skeleton");
            stripPrefix("rig");

            // normalize left/right tokens (after stripping punctuation)
            // common patterns become "l" / "r"
            auto replaceAll = [&](std::string_view a, std::string_view b) {
                for (;;) {
                    auto pos = s.find(std::string(a));
                    if (pos == std::string::npos) break;
                    s.replace(pos, a.size(), std::string(b));
                }
            };
            replaceAll("left", "l");
            replaceAll("right", "r");
            replaceAll("lft", "l");
            replaceAll("rgt", "r");

            return s;
        }

        static glm::quat EulerDegToQuat(glm::vec3 const & eulerDegXYZ) {
            glm::vec3 r = glm::radians(eulerDegXYZ);
            glm::quat qx = glm::angleAxis(r.x, glm::vec3(1,0,0));
            glm::quat qy = glm::angleAxis(r.y, glm::vec3(0,1,0));
            glm::quat qz = glm::angleAxis(r.z, glm::vec3(0,0,1));
            // XYZ order
            return glm::normalize(qz * qy * qx);
        }

        static glm::quat RotationFromTo(glm::vec3 from, glm::vec3 to) {
            float const eps = 1e-6f;
            float lf = glm::length(from);
            float lt = glm::length(to);
            if (lf < eps || lt < eps) return glm::quat(1,0,0,0);
            from /= lf;
            to   /= lt;
            float d = glm::clamp(glm::dot(from, to), -1.0f, 1.0f);
            if (d > 0.999999f) return glm::quat(1,0,0,0);
            if (d < -0.999999f) {
                // 180-degree: pick any orthogonal axis
                glm::vec3 axis = glm::cross(from, glm::vec3(1,0,0));
                if (glm::length(axis) < eps) axis = glm::cross(from, glm::vec3(0,1,0));
                axis = glm::normalize(axis);
                return glm::angleAxis(glm::pi<float>(), axis);
            }
            glm::vec3 axis = glm::normalize(glm::cross(from, to));
            float ang = std::acos(d);
            return glm::angleAxis(ang, axis);
        }

        // Build a camera ray from the current ImGui window + mouse position.
        // mousePos is already in screen space.
        static bool buildRayFromMouse(
            Engine::Camera const & cam,
            std::pair<std::uint32_t, std::uint32_t> const size,
            glm::vec3 & outOrigin,
            glm::vec3 & outDir) {
            auto * window = ImGui::GetCurrentWindow();
            if (!window) return false;
            ImGuiIO const & io = ImGui::GetIO();
            ImVec2 const mp = io.MousePos;
            ImRect const rect = window->Rect();
            if (!rect.Contains(mp)) return false;

            float w = float(size.first);
            float h = float(size.second);
            if (w <= 1.f || h <= 1.f) return false;

            // mouse within window content area
            float x = (mp.x - rect.Min.x) / rect.GetWidth();
            float y = (mp.y - rect.Min.y) / rect.GetHeight();
            // NDC
            float ndcX = x * 2.f - 1.f;
            float ndcY = 1.f - y * 2.f;

            float aspect = w / h;
            glm::mat4 proj = cam.GetProjectionMatrix(aspect);
            glm::mat4 view = cam.GetViewMatrix();
            glm::mat4 invVP = glm::inverse(proj * view);

            glm::vec4 pNear = invVP * glm::vec4(ndcX, ndcY, -1.f, 1.f);
            glm::vec4 pFar  = invVP * glm::vec4(ndcX, ndcY,  1.f, 1.f);
            if (std::abs(pNear.w) < 1e-6f || std::abs(pFar.w) < 1e-6f) return false;
            glm::vec3 nearW = glm::vec3(pNear) / pNear.w;
            glm::vec3 farW  = glm::vec3(pFar) / pFar.w;

            outOrigin = nearW;
            outDir = glm::normalize(farW - nearW);
            return true;
        }
    } // namespace

    CaseBVHSkinned::CaseBVHSkinned() :
        _flatProgram(Engine::GL::UniqueProgram({
            Engine::GL::SharedShader("assets/shaders/flat.vert"),
            Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _bonesItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines),
        _gridItem(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines) {

        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

        std::snprintf(_bvhPathBuf.data(), _bvhPathBuf.size(), "%s", _bvhPath.c_str());
        std::snprintf(_gltfPathBuf.data(), _gltfPathBuf.size(), "%s", _gltfPath.c_str());

        // Ground grid
        std::vector<glm::vec3> gridVerts;
        buildGrid(gridVerts, 10, 0.25f);
        _gridItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(gridVerts));
        _gridItem.UpdateElementBuffer(buildGridIndices(gridVerts.size() / 2));

        loadCurrentBVH();
        loadCurrentGLTF();
        rebuildRetargetMapping();
        rebuildRetargetData();
        resetCameraToContent();
    }

    void CaseBVHSkinned::loadCurrentBVH() {
        _bvhError.clear();
        _motion.reset();
        _time = 0.f;
        _framePreview = 0;
        _bvhBindLocalRot.clear();
        _bvhBindRootPos = glm::vec3(0.0f);
        _bvhBindWorldRot.clear();
        _bvhBindWorldPos.clear();
        _bvhHipNode = -1;

        std::string err;
        auto m = LoadBVH(_bvhPath, &err);
        if (!m) {
            _bvhError = err.empty() ? "Failed to load BVH." : err;
            return;
        }
        _motion = std::move(*m);

        // build BVH bone indices (for optional overlay)
        std::vector<std::uint32_t> boneIdx;
        boneIdx.reserve((_motion->Nodes.size() - 1) * 2);
        for (std::size_t i = 0; i < _motion->Nodes.size(); i++) {
            int p = _motion->Nodes[i].Parent;
            if (p >= 0) {
                boneIdx.push_back(std::uint32_t(p));
                boneIdx.push_back(std::uint32_t(i));
            }
        }
        _bonesItem.UpdateElementBuffer(boneIdx);

        computeBVHBindPose();
    }

    void CaseBVHSkinned::loadCurrentGLTF() {
        _gltfError.clear();
        _model.Clear();
        if (!_model.Load(_gltfPath)) {
            _gltfError = std::string("Failed to load glTF/GLB: ") + _gltfPath + "\n- Make sure the file exists relative to the working directory.\n- It must contain a node with both mesh + skin (skinned mesh).\n- On Windows, if your project path contains non-ASCII characters, use the memory-parse loader (see GltfSkinnedModel.cpp).";
            return;
        }
        // Reset pose to bind
        _targetLocalRot.assign(_model.Skin().JointNodeIndices.size(), glm::quat(1,0,0,0));
        _targetRootDelta = glm::vec3(0.0f);
        // Set bind-pose rotations as the initial absolute local rotations
        for (std::size_t i = 0; i < _model.Joints().size(); ++i) {
            _targetLocalRot[i] = _model.Joints()[i].BindLocalRotation;
        }
        _model.SetPose(_targetLocalRot, glm::vec3(0.0f), -1, glm::vec3(0.0f));
        _model.UpdateSkinning();
    }

    void CaseBVHSkinned::computeBVHBindPose() {
        if (!_motion.has_value()) return;
        auto const & m = *_motion;
        auto bind = SampleBVH(m, 0.f, _sampleOpt);
        if (bind.GlobalTransforms.size() != m.Nodes.size()) return;

        glm::mat4 const R = glm::mat4_cast(EulerDegToQuat(_bvhToModelEulerDeg));

        _bvhBindLocalRot.assign(m.Nodes.size(), glm::quat(1,0,0,0));
        _bvhBindWorldRot.assign(m.Nodes.size(), glm::quat(1,0,0,0));
        _bvhBindWorldPos.assign(m.Nodes.size(), glm::vec3(0.0f));

        std::vector<glm::mat4> gAdj(m.Nodes.size(), glm::mat4(1.0f));
        for (std::size_t i = 0; i < m.Nodes.size(); ++i) {
            gAdj[i] = R * bind.GlobalTransforms[i];
            _bvhBindWorldRot[i] = ExtractRotation(gAdj[i]);
            _bvhBindWorldPos[i] = ExtractTranslation(gAdj[i]);
        }

        for (std::size_t i = 0; i < m.Nodes.size(); ++i) {
            int p = m.Nodes[i].Parent;
            glm::mat4 local = gAdj[i];
            if (p >= 0) {
                local = glm::inverse(gAdj[p]) * gAdj[i];
            }
            _bvhBindLocalRot[i] = ExtractRotation(local);
        }

        _bvhBindRootPos = _bvhBindWorldPos[m.Root];
    }

    void CaseBVHSkinned::rebuildRetargetMapping() {
        _targetToBVH.clear();
        if (!_motion.has_value() || !_model.IsLoaded()) return;

        auto const & m = *_motion;
        auto const & joints = _model.Joints();
        _targetToBVH.assign(joints.size(), -1);

        // Build BVH name->index map (skip end sites)
        std::unordered_map<std::string, int> bvhByName;
        bvhByName.reserve(m.Nodes.size());
        for (int i = 0; i < int(m.Nodes.size()); ++i) {
            if (m.Nodes[i].IsEndSite) continue;
            bvhByName[normalizeName(m.Nodes[i].Name)] = i;
        }

        auto tryGet = [&](std::string const & key) -> int {
            auto it = bvhByName.find(key);
            if (it != bvhByName.end()) return it->second;
            return -1;
        };

        // A small synonym table helps with common rigs
        std::unordered_map<std::string, std::vector<std::string>> synonyms = {
            {"hips",   {"hips", "pelvis", "hip"}},
            {"spine",  {"spine", "spine1", "abdomen"}},
            {"chest",  {"chest", "spine2", "upperchest"}},
            {"neck",   {"neck"}},
            {"head",   {"head"}},
            {"lshoulder", {"lshoulder", "leftshoulder"}},
            {"rshoulder", {"rshoulder", "rightshoulder"}},
            {"larm", {"larm", "leftarm", "lupperarm", "leftupperarm"}},
            {"rarm", {"rarm", "rightarm", "rupperarm", "rightupperarm"}},
            {"lforearm", {"lforearm", "leftforearm", "llowerarm", "leftlowerarm"}},
            {"rforearm", {"rforearm", "rightforearm", "rlowerarm", "rightlowerarm"}},
            {"lhand", {"lhand", "lefthand", "lwrist", "leftwrist"}},
            {"rhand", {"rhand", "righthand", "rwrist", "rightwrist"}},
            {"lthigh", {"lthigh", "leftupleg", "lupperleg"}},
            {"rthigh", {"rthigh", "rightupleg", "rupperleg"}},
            {"lshin",  {"lshin", "leftleg", "llowerleg"}},
            {"rshin",  {"rshin", "rightleg", "rlowerleg"}},
            {"lfoot",  {"lfoot", "leftfoot", "lankle", "leftankle"}},
            {"rfoot",  {"rfoot", "rightfoot", "rankle", "rightankle"}},
        };

        for (std::size_t j = 0; j < joints.size(); ++j) {
            std::string key = normalizeName(joints[j].Name);
            int idx = tryGet(key);

            // Try suffix/prefix matches for names like "spine1" vs "spine"
            if (idx < 0) {
                for (auto const & [bvhName, bvhIdx] : bvhByName) {
                    if (bvhName == key) continue;
                    if (bvhName.size() >= 4 && (bvhName.find(key) != std::string::npos || key.find(bvhName) != std::string::npos)) {
                        idx = bvhIdx;
                        break;
                    }
                }
            }

            // Try synonym buckets
            if (idx < 0) {
                for (auto const & [bucket, keys] : synonyms) {
                    if (key.find(bucket) == std::string::npos) continue;
                    for (auto const & k : keys) {
                        idx = tryGet(k);
                        if (idx >= 0) break;
                    }
                    if (idx >= 0) break;
                }
            }

            _targetToBVH[j] = idx;
        }
    }

    void CaseBVHSkinned::rebuildRetargetData() {
        _jointOffsets.clear();
        _targetBindLocalUsed.clear();
        _targetBindWorldRotUsed.clear();
        _jointFrameCorr.clear();
        _targetHipJoint = -1;
        _bvhHipNode = -1;
        _targetRootJoint = -1;
        _autoScale = 1.0f;

        if (!_motion.has_value() || !_model.IsLoaded()) return;
        if (_targetToBVH.empty()) return;
        if (_bvhBindWorldPos.empty() || _bvhBindWorldRot.empty()) {
            // Need bind world for auto-scale / offsets.
            computeBVHBindPose();
        }
        if (_bvhBindWorldPos.empty() || _bvhBindWorldRot.empty()) return;

        auto const & m = *_motion;
        auto const & joints = _model.Joints();
        if (joints.empty()) return;

        // Target root joint (skin order)
        for (int j = 0; j < int(joints.size()); ++j) {
            if (joints[j].ParentJoint == -1) {
                _targetRootJoint = j;
                break;
            }
        }
        if (_targetRootJoint < 0) _targetRootJoint = 0;

        auto findBvhNode = [&](std::initializer_list<std::string_view> tokens) -> int {
            for (int i = 0; i < int(m.Nodes.size()); ++i) {
                if (m.Nodes[i].IsEndSite) continue;
                std::string n = normalizeName(m.Nodes[i].Name);
                for (auto t : tokens) {
                    if (n == t || n.find(std::string(t)) != std::string::npos) return i;
                }
            }
            return -1;
        };

        auto findTgtJoint = [&](std::initializer_list<std::string_view> tokens) -> int {
            for (int j = 0; j < int(joints.size()); ++j) {
                std::string n = normalizeName(joints[j].Name);
                for (auto t : tokens) {
                    if (n == t || n.find(std::string(t)) != std::string::npos) return j;
                }
            }
            return -1;
        };

        _bvhHipNode = findBvhNode({"hips", "pelvis", "hip"});
        if (_bvhHipNode < 0) _bvhHipNode = m.Root;

        _targetHipJoint = findTgtJoint({"hips", "pelvis", "hip"});
        if (_targetHipJoint < 0) _targetHipJoint = _targetRootJoint;

        // Auto scale: match approximate height (Y extent) between BVH bind and target bind.
        auto extentY = [&](auto const & positions, auto const & validFn) -> std::pair<float, float> {
            float mn = std::numeric_limits<float>::infinity();
            float mx = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int i = 0; i < int(positions.size()); ++i) {
                if (!validFn(i)) continue;
                mn = std::min(mn, positions[i].y);
                mx = std::max(mx, positions[i].y);
                any = true;
            }
            if (!any) return {0.0f, 0.0f};
            return {mn, mx};
        };

        auto [bvhMinY, bvhMaxY] = extentY(_bvhBindWorldPos, [&](int i) { return !m.Nodes[i].IsEndSite; });
        float bvhH = bvhMaxY - bvhMinY;

        std::vector<glm::vec3> tgtBindPos;
        tgtBindPos.resize(joints.size(), glm::vec3(0.0f));
        for (int j = 0; j < int(joints.size()); ++j) tgtBindPos[j] = _model.JointBindWorldPosition(j);
        auto [tgtMinY, tgtMaxY] = extentY(tgtBindPos, [&](int) { return true; });
        float tgtH = tgtMaxY - tgtMinY;

        if (bvhH > 1e-4f) _autoScale = tgtH / bvhH;
        else _autoScale = 1.0f;
        // Target bind pose used for retarget:
        // - default: glTF bind pose
        // - optional fallback: direction-based offsets (can help some rigs, but can also be unstable)
        _jointOffsets.assign(joints.size(), glm::quat(1,0,0,0));
        _targetBindLocalUsed.assign(joints.size(), glm::quat(1,0,0,0));
        for (int j = 0; j < int(joints.size()); ++j) _targetBindLocalUsed[j] = joints[j].BindLocalRotation;

        if (_useDirectionOffsets) {
            auto pickTargetChild = [&](int jointIndex) -> int {
                int best = -1;
                float bestLen2 = 0.0f;
                glm::vec3 p0 = _model.JointBindWorldPosition(jointIndex);
                for (int j = 0; j < int(joints.size()); ++j) {
                    if (joints[j].ParentJoint != jointIndex) continue;
                    glm::vec3 p1 = _model.JointBindWorldPosition(j);
                    glm::vec3 d = p1 - p0;
                    float len2 = glm::dot(d, d);
                    if (len2 > bestLen2) {
                        bestLen2 = len2;
                        best = j;
                    }
                }
                return best;
            };

            auto pickBvhChild = [&](int bvhIndex) -> int {
                int best = -1;
                float bestLen2 = 0.0f;
                glm::vec3 p0 = _bvhBindWorldPos[bvhIndex];
                for (int i = 0; i < int(m.Nodes.size()); ++i) {
                    if (m.Nodes[i].Parent != bvhIndex) continue;
                    if (m.Nodes[i].IsEndSite) continue;
                    glm::vec3 p1 = _bvhBindWorldPos[i];
                    glm::vec3 d = p1 - p0;
                    float len2 = glm::dot(d, d);
                    if (len2 > bestLen2) {
                        bestLen2 = len2;
                        best = i;
                    }
                }
                return best;
            };

            for (int j = 0; j < int(joints.size()); ++j) {
                int b = (j < int(_targetToBVH.size())) ? _targetToBVH[j] : -1;
                if (b < 0 || b >= int(m.Nodes.size())) continue;

                int childJ = pickTargetChild(j);
                int childB = pickBvhChild(b);
                if (childJ < 0 || childB < 0) continue;

                glm::vec3 pT0 = _model.JointBindWorldPosition(j);
                glm::vec3 pT1 = _model.JointBindWorldPosition(childJ);
                glm::vec3 pS0 = _bvhBindWorldPos[b];
                glm::vec3 pS1 = _bvhBindWorldPos[childB];

                glm::vec3 dirT = pT1 - pT0;
                glm::vec3 dirS = pS1 - pS0;
                if (glm::length(dirT) < 1e-5f || glm::length(dirS) < 1e-5f) continue;
                dirT = glm::normalize(dirT);
                dirS = glm::normalize(dirS);

                // Convert world directions into target joint local space (bind pose).
                glm::quat bindWorld = joints[j].BindWorldRotation;
                glm::vec3 localT = glm::normalize(glm::inverse(bindWorld) * dirT);
                glm::vec3 localS = glm::normalize(glm::inverse(bindWorld) * dirS);
                _jointOffsets[j] = glm::normalize(RotationFromTo(localT, localS));
            }

            for (int j = 0; j < int(joints.size()); ++j) {
                _targetBindLocalUsed[j] = glm::normalize(joints[j].BindLocalRotation * _jointOffsets[j]);
            }
        }

        // Compute bind world rotations for the chosen target bind (hierarchical accumulation).
        _targetBindWorldRotUsed.assign(joints.size(), glm::quat(1,0,0,0));
        std::vector<std::uint8_t> done(joints.size(), 0);
        std::function<glm::quat(int)> solve = [&](int j) -> glm::quat {
            if (j < 0 || j >= int(joints.size())) return glm::quat(1,0,0,0);
            if (done[j]) return _targetBindWorldRotUsed[j];
            int p = joints[j].ParentJoint;
            glm::quat parentW = (p >= 0) ? solve(p) : _model.JointExternalParentBindWorldRotation(j);
            _targetBindWorldRotUsed[j] = glm::normalize(parentW * _targetBindLocalUsed[j]);
            done[j] = 1;
            return _targetBindWorldRotUsed[j];
        };
        for (int j = 0; j < int(joints.size()); ++j) (void)solve(j);

        // Per-joint frame correction: maps target bind-local basis -> BVH bind-local basis.
        _jointFrameCorr.assign(joints.size(), glm::quat(1,0,0,0));
        for (int j = 0; j < int(joints.size()); ++j) {
            int b = (j < int(_targetToBVH.size())) ? _targetToBVH[j] : -1;
            if (b < 0 || b >= int(m.Nodes.size())) continue;
            _jointFrameCorr[j] = glm::normalize(_bvhBindLocalRot[b] * glm::inverse(_targetBindLocalUsed[j]));
        }
    }

    void CaseBVHSkinned::retargetAtTime(float tSeconds) {
        if (!_motion.has_value() || !_model.IsLoaded()) return;
        auto const & m = *_motion;

        auto sample = SampleBVH(m, tSeconds, _sampleOpt);
        if (sample.GlobalTransforms.size() != m.Nodes.size()) return;

        // Ensure bind / offsets caches exist
        if (_bvhBindWorldRot.size() != m.Nodes.size() || _bvhBindWorldPos.size() != m.Nodes.size()) {
            computeBVHBindPose();
        }
        if (_bvhBindWorldRot.size() != m.Nodes.size() || _bvhBindWorldPos.size() != m.Nodes.size()) return;

        auto const & joints = _model.Joints();
        if (_targetBindLocalUsed.size() != joints.size() ||
            _jointFrameCorr.size() != joints.size() ||
            _targetBindWorldRotUsed.size() != joints.size()) {
            rebuildRetargetData();
        }
        if (_targetBindLocalUsed.size() != joints.size() || _jointFrameCorr.size() != joints.size()) return;

        glm::mat4 const R = glm::mat4_cast(EulerDegToQuat(_bvhToModelEulerDeg));
        float const effectiveScale = _userScale * (_autoScaleEnabled ? _autoScale : 1.0f);

        // BVH pose at t (after global adjustment)
        std::vector<glm::mat4> gAdj(m.Nodes.size(), glm::mat4(1.0f));
        std::vector<glm::quat> bvhLocalRot(m.Nodes.size(), glm::quat(1,0,0,0));
        std::vector<glm::vec3> bvhWorldPos(m.Nodes.size(), glm::vec3(0.0f));
        for (std::size_t i = 0; i < m.Nodes.size(); ++i) {
            gAdj[i] = R * sample.GlobalTransforms[i];
            bvhWorldPos[i] = ExtractTranslation(gAdj[i]);
        }
        for (std::size_t i = 0; i < m.Nodes.size(); ++i) {
            int p = m.Nodes[i].Parent;
            glm::mat4 local = gAdj[i];
            if (p >= 0) local = glm::inverse(gAdj[p]) * gAdj[i];
            bvhLocalRot[i] = ExtractRotation(local);
        }

        // Root / hip translation deltas (world)
        glm::vec3 hipDeltaWorld(0.0f);
        int hipSrc = (_bvhHipNode >= 0 && _bvhHipNode < int(m.Nodes.size())) ? _bvhHipNode : m.Root;
        hipDeltaWorld = (bvhWorldPos[hipSrc] - _bvhBindWorldPos[hipSrc]) * effectiveScale;
        if (!_useRootMotion) hipDeltaWorld = glm::vec3(0.0f);

        glm::vec3 rootDeltaWorld(0.0f);
        glm::vec3 hipDeltaWorldApplied(0.0f);
        switch (_hipMode) {
        case HipMode::Root:
            rootDeltaWorld = hipDeltaWorld;
            hipDeltaWorldApplied = glm::vec3(0.0f);
            break;
        case HipMode::Hip:
            rootDeltaWorld = glm::vec3(0.0f);
            hipDeltaWorldApplied = hipDeltaWorld;
            break;
        case HipMode::SplitXZRoot_YHip:
        default:
            rootDeltaWorld = glm::vec3(hipDeltaWorld.x, 0.0f, hipDeltaWorld.z);
            hipDeltaWorldApplied = glm::vec3(0.0f, hipDeltaWorld.y, 0.0f);
            break;
        }

        if (_inPlaceTarget) {
            rootDeltaWorld.x = rootDeltaWorld.z = 0.0f;
            hipDeltaWorldApplied.x = hipDeltaWorldApplied.z = 0.0f;
        }
        if (_preserveHipXZ) {
            hipDeltaWorldApplied.x = hipDeltaWorldApplied.z = 0.0f;
        }

        // A/B/C: local delta + change-of-basis (like three.js retargeting)
        _targetLocalRot.assign(joints.size(), glm::quat(1,0,0,0));
        for (int j = 0; j < int(joints.size()); ++j) {
            int bvhIdx = (j < int(_targetToBVH.size())) ? _targetToBVH[j] : -1;
            if (bvhIdx >= 0 && bvhIdx < int(m.Nodes.size())) {
                glm::quat deltaS = glm::normalize(bvhLocalRot[bvhIdx] * glm::inverse(_bvhBindLocalRot[bvhIdx]));
                glm::quat C = _jointFrameCorr[j];
                glm::quat deltaT = glm::normalize(glm::inverse(C) * deltaS * C);
                _targetLocalRot[j] = glm::normalize(deltaT * _targetBindLocalUsed[j]);
            } else {
                _targetLocalRot[j] = _targetBindLocalUsed[j];
            }
        }

        // Compute pose world rotations (needed to convert hip translation into parent local space)
        std::vector<glm::quat> tgtWorldPose(joints.size(), glm::quat(1,0,0,0));
        std::vector<std::uint8_t> donePose(joints.size(), 0);
        std::function<glm::quat(int)> solvePose = [&](int j) -> glm::quat {
            if (j < 0 || j >= int(joints.size())) return glm::quat(1,0,0,0);
            if (donePose[j]) return tgtWorldPose[j];
            int p = joints[j].ParentJoint;
            glm::quat parentW = (p >= 0) ? solvePose(p) : _model.JointExternalParentBindWorldRotation(j);
            tgtWorldPose[j] = glm::normalize(parentW * _targetLocalRot[j]);
            donePose[j] = 1;
            return tgtWorldPose[j];
        };
        for (int j = 0; j < int(joints.size()); ++j) (void)solvePose(j);

        // Convert translation deltas into the spaces expected by SetPose.
        _targetRootDelta = glm::vec3(0.0f);
        _targetHipDelta  = glm::vec3(0.0f);

        if (_targetRootJoint >= 0 && _targetRootJoint < int(joints.size())) {
            glm::mat4 parentBind = _model.JointExternalParentBindWorldMatrix(_targetRootJoint);
            _targetRootDelta = glm::vec3(glm::inverse(parentBind) * glm::vec4(rootDeltaWorld, 0.0f));
        }

        if (_targetHipJoint >= 0 && _targetHipJoint < int(joints.size())) {
            int hipParent = joints[_targetHipJoint].ParentJoint;
            if (hipParent < 0) {
                glm::mat4 parentBind = _model.JointExternalParentBindWorldMatrix(_targetHipJoint);
                _targetHipDelta = glm::vec3(glm::inverse(parentBind) * glm::vec4(hipDeltaWorldApplied, 0.0f));
            } else {
                _targetHipDelta = glm::inverse(solvePose(hipParent)) * hipDeltaWorldApplied;
            }
        }

        _model.SetPose(_targetLocalRot, _targetRootDelta, _targetHipJoint, _targetHipDelta);
        _model.UpdateSkinning();

        if (_showBVHSkeleton) {
            _bvhPosePositions.resize(m.Nodes.size());

            // Align the BVH overlay to the animated skinned character so they can be compared directly.
            // We anchor on the (retargeted) hip joint in the target skeleton.
            glm::vec3 alignOffset(0.0f);
            if (_targetHipJoint >= 0 && _targetHipJoint < int(_model.Skin().JointWorldMatrices.size()) &&
                hipSrc >= 0 && hipSrc < int(m.Nodes.size())) {
                glm::vec3 modelHipPos = glm::vec3(_model.Skin().JointWorldMatrices[_targetHipJoint][3]);
                alignOffset = modelHipPos - bvhWorldPos[hipSrc] * effectiveScale;
            }

            for (int i = 0; i < int(m.Nodes.size()); ++i) {
                _bvhPosePositions[i] = bvhWorldPos[i] * effectiveScale + alignOffset;
            }
            _bonesItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_bvhPosePositions));
        }
    }

    void CaseBVHSkinned::resetCameraToContent() {
        // Prefer model bounds (via a CPU skinned sample at bind pose); fallback to BVH.
        glm::vec3 mn( std::numeric_limits<float>::infinity());
        glm::vec3 mx(-std::numeric_limits<float>::infinity());
        bool has = false;

        if (_model.IsLoaded()) {
            std::vector<glm::vec3> pts;
            _model.SkinPositionsCPU(pts, nullptr);
            if (!pts.empty()) {
                for (auto const & p : pts) {
                    mn = glm::min(mn, p);
                    mx = glm::max(mx, p);
                }
                has = true;
            }
        }

        if (!has && _motion.has_value()) {
            auto pose = SampleBVH(*_motion, 0.f, _sampleOpt);
            if (!pose.JointPositions.empty()) {
                for (auto const & p : pose.JointPositions) {
                    mn = glm::min(mn, p);
                    mx = glm::max(mx, p);
                }
                has = true;
            }
        }

        if (!has) return;

        glm::vec3 c = 0.5f * (mn + mx);
        float radius = glm::length(mx - mn) * 0.6f;
        radius = std::max(radius, 0.5f);

        _camera.Target = c + glm::vec3(0, radius * 0.3f, 0);
        _camera.Eye    = _camera.Target + glm::vec3(-radius * 2.2f, radius * 1.6f, radius * 2.2f);
        _cameraManager.Save(_camera);
    }

    void CaseBVHSkinned::OnSetupPropsUI() {
        ImGui::Text("BVH -> glTF Skinned Retarget");
        ImGui::Separator();

        if (ImGui::CollapsingHeader("Files", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Paths are relative to the executable working directory.");

            // Motion presets (use the same set as Case 1)
            {
                struct Preset {
                    char const * Label;
                    char const * Path;
                };
                static Preset const kPresets[] = {
                    {"T-pose (bind)",         "assets/motions/T-pose.bvh"},                    {"Jog",                   "assets/motions/jog.bvh"},
                    {"Running",               "assets/motions/running.bvh"},
                    {"Walking",               "assets/motions/walking.bvh"},
                    {"Start Walking",         "assets/motions/start walking.bvh"},
                    {"Jump",                  "assets/motions/jump.bvh"},
                };

                int preset = _motionPreset;
                if (ImGui::Combo(
                        "Motion Preset", &preset,
                        [](void * data, int idx, char const ** out_text) {
                            auto const * p = reinterpret_cast<Preset const *>(data);
                            *out_text = p[idx].Label;
                            return true;
                        },
                        (void *)kPresets, int(sizeof(kPresets) / sizeof(kPresets[0])))) {
                    _motionPreset = preset;
                    _bvhPath = kPresets[_motionPreset].Path;
                    std::snprintf(_bvhPathBuf.data(), _bvhPathBuf.size(), "%s", _bvhPath.c_str());

                    loadCurrentBVH();
                    rebuildRetargetMapping();
                    rebuildRetargetData();
                    resetCameraToContent();
                    _time = 0.0f;
                    _framePreview = 0;
                }
            }

            ImGui::InputText("BVH Path", _bvhPathBuf.data(), _bvhPathBuf.size());
            if (ImGui::Button("Load BVH")) {
                _bvhPath = std::string(_bvhPathBuf.data());
                loadCurrentBVH();
                rebuildRetargetMapping();
                rebuildRetargetData();
                resetCameraToContent();
            }
            if (!_bvhError.empty()) {
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "BVH Error: %s", _bvhError.c_str());
            }

            ImGui::Spacing();
            ImGui::InputText("glTF/GLB Path", _gltfPathBuf.data(), _gltfPathBuf.size());
            if (ImGui::Button("Load glTF")) {
                _gltfPath = std::string(_gltfPathBuf.data());
                loadCurrentGLTF();
                rebuildRetargetMapping();
                rebuildRetargetData();
                resetCameraToContent();
            }
            if (!_gltfError.empty()) {
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "glTF Error: %s", _gltfError.c_str());
            }

            if (_motion.has_value()) {
                auto const & m = *_motion;
                ImGui::Spacing();
                ImGui::Text("BVH Nodes: %d", int(m.Nodes.size()));
                ImGui::Text("BVH Frames: %d", m.NumFrames);
                ImGui::Text("BVH FPS: %.1f", 1.f / m.FrameTime);
            }
            if (_model.IsLoaded()) {
                ImGui::Spacing();
                ImGui::Text("glTF Joints (skin): %d", int(_model.Joints().size()));
            }

            if (ImGui::Button("Rebuild Retarget Mapping")) {
                rebuildRetargetMapping();
                rebuildRetargetData();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Camera")) resetCameraToContent();
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

        if (ImGui::CollapsingHeader("Sampling / Retarget", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool changedSampling = false;
            changedSampling |= ImGui::SliderFloat("BVH Scale", &_sampleOpt.Scale, 0.001f, 0.1f, "%.4f");
            changedSampling |= ImGui::Checkbox("BVH In Place (zero root XZ)", &_sampleOpt.InPlace);
            changedSampling |= ImGui::Checkbox("BVH Z-up -> Y-up (-90deg X)", &_sampleOpt.RotateZUpToYUp);

            bool changedEuler = ImGui::SliderFloat3(
                "BVH->Model Euler (deg XYZ)",
                glm::value_ptr(_bvhToModelEulerDeg),
                -180.f,
                180.f,
                "%.1f");

            if (changedSampling || changedEuler) {
                computeBVHBindPose();
                rebuildRetargetData();
            }

            if (ImGui::Button("Recompute BVH Bind")) {
                computeBVHBindPose();
                rebuildRetargetData();
            }

            ImGui::Separator();
            ImGui::Checkbox("Use BVH Root Motion", &_useRootMotion);
            ImGui::Checkbox("Force Target In Place", &_inPlaceTarget);

            ImGui::Checkbox("Auto Scale", &_autoScaleEnabled);            ImGui::SliderFloat("User Scale", &_userScale, 0.05f, 5.0f, "%.3f");
            ImGui::TextDisabled("Computed Auto Scale: %.4f", _autoScale);
            ImGui::TextDisabled("Effective Scale (User*Auto): %.4f", _userScale * (_autoScaleEnabled ? _autoScale : 1.0f));

            // Hip mode
            {
                const char * modes[] = {"Root", "Hip", "Split XZ (Root) + Y (Hip)"};
                int mode = int(_hipMode);
                if (ImGui::Combo("Hip Mode", &mode, modes, IM_ARRAYSIZE(modes))) {
                    _hipMode = HipMode(mode);
                }
            }
            ImGui::Checkbox("Preserve Hip XZ (keep only vertical hip motion)", &_preserveHipXZ);
            ImGui::TextDisabled("Tip: many rigs need BVH->Model Yaw=180 to face forward.");
        }

        if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Show Grid", &_showGrid);
            ImGui::Checkbox("Show Mesh", &_showMesh);
            ImGui::Checkbox("Show BVH Skeleton Overlay", &_showBVHSkeleton);
            ImGui::SliderFloat("Bone Width", &_boneWidth, 0.5f, 4.f);
            ImGui::ColorEdit4("Mesh Color", glm::value_ptr(_meshColor));
        }

        if (ImGui::CollapsingHeader("CPU (On-demand) Picking", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable CPU Raycast (click left)", &_enablePicking);
            if (_hasHit) {
                ImGui::Text("Hit T = %.4f", _hit.T);
                ImGui::Text("Hit Tri = %u", _hit.TriangleIndex);
                ImGui::Text("Hit Pos = (%.3f, %.3f, %.3f)", _hit.Position.x, _hit.Position.y, _hit.Position.z);
            } else {
                ImGui::TextDisabled("No hit yet.");
            }
        }
    }
    Common::CaseRenderResult CaseBVHSkinned::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        _lastRenderSize = desiredSize;
        _frame.Resize(desiredSize);

        _cameraManager.Update(_camera);
        float aspect = (float(desiredSize.first) / desiredSize.second);
        glm::mat4 proj = _camera.GetProjectionMatrix(aspect);
        glm::mat4 view = _camera.GetViewMatrix();

        _flatProgram.GetUniforms().SetByName("u_Projection", proj);
        _flatProgram.GetUniforms().SetByName("u_View", view);

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_LINE_SMOOTH);

        if (_motion.has_value() && _model.IsLoaded()) {
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

            retargetAtTime(_time);
        }

        // Draw grid
        if (_showGrid) {
            glLineWidth(1.f);
            _flatProgram.GetUniforms().SetByName("u_Color", _gridColor);
            _gridItem.Draw({ _flatProgram.Use() });
        }

        // Draw BVH skeleton overlay
        if (_showBVHSkeleton && _motion.has_value()) {
            glLineWidth(_boneWidth);
            _flatProgram.GetUniforms().SetByName("u_Color", _boneColor);
            _bonesItem.Draw({ _flatProgram.Use() });
        }

        // Draw skinned mesh (GPU skinning)
        if (_showMesh && _model.IsLoaded()) {
            _model.Draw(view, proj, _meshColor);
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

    void CaseBVHSkinned::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);

        if (!_enablePicking || !_model.IsLoaded()) return;

        auto * window = ImGui::GetCurrentWindow();
        if (!window) return;
        bool anyHeld = false;
        bool hover = false;
        ImGui::ButtonBehavior(window->Rect(), window->GetID("##io"), &hover, &anyHeld);

        if (!hover) return;
        if (!ImGui::IsMouseClicked(ImGuiMouseButton_Left)) return;

        // Avoid stealing orbit rotate click when Ctrl/Shift are held (common camera input)
        ImGuiIO const & io = ImGui::GetIO();
        if (io.KeyCtrl || io.KeyShift) return;

        glm::vec3 ro, rd;
        if (!buildRayFromMouse(_camera, _lastRenderSize, ro, rd)) return;

        GltfSkinnedModel::RaycastHit hit;
        if (_model.RaycastCPU(ro, rd, hit)) {
            _hasHit = true;
            _hit = hit;
        } else {
            _hasHit = false;
        }
    }

} // namespace VCX::Labs::Animation
