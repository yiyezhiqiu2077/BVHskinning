#define CGLTF_IMPLEMENTATION
#include "Labs/4-Animation/GltfSkinnedModel.h"

#include <cassert>
#include <array>
#include <cstring>
#include <limits>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Engine/prelude.hpp"

namespace VCX::Labs::Animation {
    namespace {
        static glm::mat4 Mat4FromCGLTF(float const m[16]) {
            // glTF is column-major. glm::mat4 constructor expects column vectors.
            glm::mat4 r {1.0f};
            std::memcpy(glm::value_ptr(r), m, sizeof(float) * 16);
            return r;
        }

        static glm::quat QuatFromCGLTF(float const q[4]) {
            // cgltf stores [x,y,z,w]; glm expects (w,x,y,z)
            return glm::normalize(glm::quat(q[3], q[0], q[1], q[2]));
        }

        static glm::vec3 Vec3FromCGLTF(float const v[3]) {
            return glm::vec3(v[0], v[1], v[2]);
        }

        static glm::mat4 TRS(glm::vec3 const & t, glm::quat const & r, glm::vec3 const & s) {
            return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0f), s);
        }

        static glm::quat WorldRotationFromMat4(glm::mat4 const & w) {
            // Remove scaling before quat_cast
            glm::mat3 m(w);
            // Orthonormalize columns
            m[0] = glm::normalize(m[0]);
            m[1] = glm::normalize(m[1]);
            m[2] = glm::normalize(m[2]);
            return glm::normalize(glm::quat_cast(m));
        }

        static glm::vec3 WorldTranslationFromMat4(glm::mat4 const & w) {
            return glm::vec3(w[3]);
        }

        static bool IntersectRayTriangle(
            glm::vec3 const & ro,
            glm::vec3 const & rd,
            glm::vec3 const & v0,
            glm::vec3 const & v1,
            glm::vec3 const & v2,
            float &           outT,
            float &           outU,
            float &           outV) {
            // Möller–Trumbore
            glm::vec3 const e1 = v1 - v0;
            glm::vec3 const e2 = v2 - v0;
            glm::vec3 const p = glm::cross(rd, e2);
            float const det = glm::dot(e1, p);
            if (std::abs(det) < 1e-8f) return false;
            float const invDet = 1.0f / det;
            glm::vec3 const tvec = ro - v0;
            float const u = glm::dot(tvec, p) * invDet;
            if (u < 0.0f || u > 1.0f) return false;
            glm::vec3 const q = glm::cross(tvec, e1);
            float const v = glm::dot(rd, q) * invDet;
            if (v < 0.0f || u + v > 1.0f) return false;
            float const t = glm::dot(e2, q) * invDet;
            if (t < 0.0f) return false;
            outT = t;
            outU = u;
            outV = v;
            return true;
        }

        template<typename T>
        static T Clamp(T v, T lo, T hi) {
            return v < lo ? lo : (v > hi ? hi : v);
        }

        static std::string SafeName(char const * n) {
            return n ? std::string(n) : std::string();
        }
    } // namespace

    glm::mat4 GltfSkinnedModel::MakeLocalMatrix(cgltf_node const * n) {
        if (!n) return glm::mat4(1.0f);
        if (n->has_matrix) {
            return Mat4FromCGLTF(n->matrix);
        }
        glm::vec3 t {0.0f};
        glm::quat r {1, 0, 0, 0};
        glm::vec3 s {1.0f};
        if (n->has_translation) t = Vec3FromCGLTF(n->translation);
        if (n->has_rotation)    r = QuatFromCGLTF(n->rotation);
        if (n->has_scale)       s = Vec3FromCGLTF(n->scale);
        return TRS(t, r, s);
    }

    void GltfSkinnedModel::Clear() {
        _loaded = false;
        _sourcePath.clear();
        _nodes.clear();
        _nodesBindWorld.clear();
        _meshNode = -1;
        _meshWorld = glm::mat4(1.0f);
        _skin = {};
        _jointInfos.clear();
        _cpuVertices.clear();
        _indices.clear();
        _renderItem.reset();
        _program.reset();

        if (_boneTexture != 0) {
            glDeleteTextures(1, &_boneTexture);
            _boneTexture = 0;
        }
        _boneTextureSize = 0;
        _boneTexturePixels.clear();
        _uBoneTextureLoc = -1;
        _uBoneTextureSizeLoc = -1;

        _poseLocalRot.clear();
        _poseRootDelta = glm::vec3(0.0f);
        _poseHipJoint = -1;
        _poseHipDelta = glm::vec3(0.0f);
    }

    void GltfSkinnedModel::EnsureShaders() {
        if (_program.has_value()) return;
        _program.emplace(std::initializer_list<Engine::GL::SharedShader> {
            Engine::GL::SharedShader("assets/shaders/skinned.vert"),
            Engine::GL::SharedShader("assets/shaders/skinned.frag"),
        });
        // Cache uniform locations for bone texture
        _uBoneTextureLoc = glGetUniformLocation(_program->Get(), "u_BoneTexture");
        _uBoneTextureSizeLoc = glGetUniformLocation(_program->Get(), "u_BoneTextureSize");
}

    void GltfSkinnedModel::EnsureRenderItem() {
        if (_renderItem.has_value()) return;
        EnsureShaders();

        Engine::GL::VertexLayout const layout =
            Engine::GL::VertexLayout()
                .Add<SkinnedVertex>("vertex", Engine::GL::DrawFrequency::Static)
                .At<SkinnedVertex>(0, &SkinnedVertex::Position)
                .At<SkinnedVertex>(1, &SkinnedVertex::Normal)
                .At<SkinnedVertex>(2, &SkinnedVertex::TexCoord)
                .At<SkinnedVertex>(3, &SkinnedVertex::SkinIndex)
                .At<SkinnedVertex>(4, &SkinnedVertex::SkinWeight);

        _renderItem.emplace(layout, Engine::GL::PrimitiveType::Triangles);
        _renderItem->UpdateVertexBuffer("vertex", Engine::make_span_bytes<SkinnedVertex>(_cpuVertices));
        _renderItem->UpdateElementBuffer(_indices);
    }


void GltfSkinnedModel::EnsureBoneTexture() {
    if (!_loaded) return;
    int const boneCount = int(_skin.BoneMatrices.size());
    if (boneCount <= 0) return;

    int const requiredTexels = boneCount * 4; // 4 vec4 columns per mat4
    int size = int(std::ceil(std::sqrt(float(requiredTexels))));
    if (size <= 0) size = 1;

    // Recreate if size changed
    if (_boneTexture != 0 && _boneTextureSize == size) return;

    if (_boneTexture != 0) {
        glDeleteTextures(1, &_boneTexture);
        _boneTexture = 0;
    }

    _boneTextureSize = size;
    _boneTexturePixels.assign(size * size * 4, 0.0f);

    glGenTextures(1, &_boneTexture);
    glBindTexture(GL_TEXTURE_2D, _boneTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA32F,
        _boneTextureSize, _boneTextureSize,
        0, GL_RGBA, GL_FLOAT,
        _boneTexturePixels.data());

    glBindTexture(GL_TEXTURE_2D, 0);
}

void GltfSkinnedModel::UpdateBoneTexture() {
    if (_boneTexture == 0) return;
    int const boneCount = int(_skin.BoneMatrices.size());
    if (boneCount <= 0) return;
    int const size = _boneTextureSize;
    if (size <= 0) return;

    // Write bone matrices into pixel buffer (column-major: 4 texels = 4 columns)
    // texel(base + c) stores column c (vec4) of the matrix.
    for (int i = 0; i < boneCount; ++i) {
        glm::mat4 const & m = _skin.BoneMatrices[i];
        float const * p = glm::value_ptr(m); // 16 floats, column-major
        int baseTexel = i * 4;

        for (int c = 0; c < 4; ++c) {
            int texel = baseTexel + c;
            int x = texel % size;
            int y = texel / size;
            if (y >= size) continue; // safety

            int pixelBase = (y * size + x) * 4;
            _boneTexturePixels[pixelBase + 0] = p[c * 4 + 0];
            _boneTexturePixels[pixelBase + 1] = p[c * 4 + 1];
            _boneTexturePixels[pixelBase + 2] = p[c * 4 + 2];
            _boneTexturePixels[pixelBase + 3] = p[c * 4 + 3];
        }
    }

    glBindTexture(GL_TEXTURE_2D, _boneTexture);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        size, size,
        GL_RGBA, GL_FLOAT,
        _boneTexturePixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}


    void GltfSkinnedModel::BuildNodes(cgltf_data const * data) {
        _nodes.clear();
        _nodes.resize(data->nodes_count);

        auto indexOf = [&](cgltf_node const * p) -> int {
            return p ? int(p - data->nodes) : -1;
        };

        for (cgltf_size i = 0; i < data->nodes_count; ++i) {
            cgltf_node const * n = &data->nodes[i];
            Node node;
            node.Name = SafeName(n->name);
            node.LocalBind = MakeLocalMatrix(n);
            node.Local = node.LocalBind;

            // Try to recover bind TRS for later pose overwrite. If matrix was provided, we approximate
            // translation from the matrix and assume uniform scale if any.
            node.BindTranslation = glm::vec3(node.LocalBind[3]);
            node.BindRotation = WorldRotationFromMat4(node.LocalBind);
            // Scale: length of basis vectors
            node.BindScale = glm::vec3(
                glm::length(glm::vec3(node.LocalBind[0])),
                glm::length(glm::vec3(node.LocalBind[1])),
                glm::length(glm::vec3(node.LocalBind[2])));

            _nodes[i] = std::move(node);
        }

        // Parent/child relationships
        for (cgltf_size i = 0; i < data->nodes_count; ++i) {
            cgltf_node const * n = &data->nodes[i];
            for (cgltf_size c = 0; c < n->children_count; ++c) {
                int child = indexOf(n->children[c]);
                if (child >= 0) {
                    _nodes[child].Parent = int(i);
                    _nodes[i].Children.push_back(child);
                }
            }
        }
    }

    void GltfSkinnedModel::ComputeWorldMatrices() {
        // Roots are nodes with Parent == -1
        std::vector<int> stack;
        stack.reserve(_nodes.size());
        for (int i = 0; i < int(_nodes.size()); ++i) {
            if (_nodes[i].Parent == -1) stack.push_back(i);
        }

        // DFS stack: compute world, then push children
        while (!stack.empty()) {
            int n = stack.back();
            stack.pop_back();
            int p = _nodes[n].Parent;
            if (p == -1) _nodes[n].World = _nodes[n].Local;
            else         _nodes[n].World = _nodes[p].World * _nodes[n].Local;
            for (int ch : _nodes[n].Children) stack.push_back(ch);
        }

        if (_meshNode >= 0) {
            _meshWorld = _nodes[_meshNode].World;
        }
    }

    bool GltfSkinnedModel::BuildSkin(cgltf_data const * data, cgltf_skin const * skin, cgltf_node const * meshNode) {
        if (!skin) return false;

        auto indexOf = [&](cgltf_node const * p) -> int { return p ? int(p - data->nodes) : -1; };

        _skin = {};
        _skin.JointNodeIndices.reserve(skin->joints_count);
        _skin.InverseBindMatrices.reserve(skin->joints_count);
        _skin.JointWorldMatrices.resize(skin->joints_count, glm::mat4(1.0f));
        _skin.BoneMatrices.resize(skin->joints_count, glm::mat4(1.0f));
        _skin.JointParent.resize(skin->joints_count, -1);

        for (cgltf_size j = 0; j < skin->joints_count; ++j) {
            int nodeIndex = indexOf(skin->joints[j]);
            _skin.JointNodeIndices.push_back(nodeIndex);
            _skin.NodeToJoint[nodeIndex] = int(j);
        }

        // Parent joints in skin order: walk glTF node parents until hitting another joint.
        for (int j = 0; j < int(_skin.JointNodeIndices.size()); ++j) {
            int nodeIndex = _skin.JointNodeIndices[j];
            int p = _nodes[nodeIndex].Parent;
            int parentJoint = -1;
            while (p != -1) {
                if (auto it = _skin.NodeToJoint.find(p); it != _skin.NodeToJoint.end()) {
                    parentJoint = it->second;
                    break;
                }
                p = _nodes[p].Parent;
            }
            _skin.JointParent[j] = parentJoint;
        }

        // Inverse bind matrices
        if (skin->inverse_bind_matrices) {
            auto const * acc = skin->inverse_bind_matrices;
            if (acc->type != cgltf_type_mat4) return false;
            for (cgltf_size i = 0; i < acc->count; ++i) {
                float m[16];
                cgltf_accessor_read_float(acc, i, m, 16);
                _skin.InverseBindMatrices.push_back(Mat4FromCGLTF(m));
            }
        } else {
            // Fallback: identity
            _skin.InverseBindMatrices.resize(_skin.JointNodeIndices.size(), glm::mat4(1.0f));
        }

        // Bind matrix is the mesh node world at bind time (like three.js bind())
        if (meshNode) {
            int meshIndex = indexOf(meshNode);
            if (meshIndex >= 0) {
                _skin.BindMatrix = _nodes[meshIndex].World;
                _skin.BindMatrixInverse = glm::inverse(_skin.BindMatrix);
            }
        }

        // IMPORTANT (glTF vs three.js convention):
        // glTF `inverseBindMatrices` are defined in the *mesh node's local space*.
        // three.js (and our shader) expects boneInverses in *world space* (inverse of joint world matrix at bind).
        // Convert: ibm_world = ibm_gltf * inverse(bindMatrix).
        // This prevents the bindMatrix from being applied twice in the skinning formula (which would shrink the mesh
        // dramatically when the mesh has a scale like 0.01).
        if (!_skin.InverseBindMatrices.empty()) {
            for (auto & ibm : _skin.InverseBindMatrices) {
                ibm = ibm * _skin.BindMatrixInverse;
            }
        }

        // Joint info (bind pose) for retargeting
        _jointInfos.clear();
        _jointInfos.resize(_skin.JointNodeIndices.size());
        for (int j = 0; j < int(_skin.JointNodeIndices.size()); ++j) {
            int nodeIndex = _skin.JointNodeIndices[j];
            _jointInfos[j].Name = _nodes[nodeIndex].Name;
            _jointInfos[j].ParentJoint = _skin.JointParent[j];
            _jointInfos[j].NodeIndex = nodeIndex;
            _jointInfos[j].BindLocalTranslation = _nodes[nodeIndex].BindTranslation;
            _jointInfos[j].BindLocalRotation = _nodes[nodeIndex].BindRotation;
            _jointInfos[j].BindWorldRotation = WorldRotationFromMat4(_nodes[nodeIndex].World);
        }

        // Initialize matrices with bind pose
        for (int j = 0; j < int(_skin.JointNodeIndices.size()); ++j) {
            int nodeIndex = _skin.JointNodeIndices[j];
            _skin.JointWorldMatrices[j] = _nodes[nodeIndex].World;
            _skin.BoneMatrices[j] = _skin.JointWorldMatrices[j] * _skin.InverseBindMatrices[j];
        }
        return true;
    }

    bool GltfSkinnedModel::BuildGeometry(cgltf_data const * data, cgltf_mesh const * mesh) {
        if (!mesh || mesh->primitives_count == 0) return false;

        // Choose first triangle primitive
        cgltf_primitive const * prim = nullptr;
        for (cgltf_size i = 0; i < mesh->primitives_count; ++i) {
            if (mesh->primitives[i].type == cgltf_primitive_type_triangles) {
                prim = &mesh->primitives[i];
                break;
            }
        }
        if (!prim) return false;

        cgltf_accessor const * aPos = nullptr;
        cgltf_accessor const * aNor = nullptr;
        cgltf_accessor const * aUV0 = nullptr;
        cgltf_accessor const * aJnt = nullptr;
        cgltf_accessor const * aWgt = nullptr;

        for (cgltf_size i = 0; i < prim->attributes_count; ++i) {
            auto const & attr = prim->attributes[i];
            switch (attr.type) {
            case cgltf_attribute_type_position: aPos = attr.data; break;
            case cgltf_attribute_type_normal: aNor = attr.data; break;
            case cgltf_attribute_type_texcoord:
                if (attr.index == 0) aUV0 = attr.data;
                break;
            case cgltf_attribute_type_joints:
                if (attr.index == 0) aJnt = attr.data;
                break;
            case cgltf_attribute_type_weights:
                if (attr.index == 0) aWgt = attr.data;
                break;
            default: break;
            }
        }
        if (!aPos) return false;

        cgltf_size vertexCount = aPos->count;
        _cpuVertices.resize(vertexCount);

        for (cgltf_size v = 0; v < vertexCount; ++v) {
            SkinnedVertex sv;
            float p[3] {0,0,0};
            cgltf_accessor_read_float(aPos, v, p, 3);
            sv.Position = glm::vec3(p[0], p[1], p[2]);

            if (aNor) {
                float n[3] {0,1,0};
                cgltf_accessor_read_float(aNor, v, n, 3);
                sv.Normal = glm::vec3(n[0], n[1], n[2]);
            } else {
                sv.Normal = glm::vec3(0, 1, 0);
            }

            if (aUV0) {
                float uv[2] {0,0};
                cgltf_accessor_read_float(aUV0, v, uv, 2);
                sv.TexCoord = glm::vec2(uv[0], uv[1]);
            } else {
                sv.TexCoord = glm::vec2(0, 0);
            }

            if (aJnt) {
                cgltf_uint j[4] {0,0,0,0};
                cgltf_accessor_read_uint(aJnt, v, j, 4);
                sv.SkinIndex = glm::vec4(float(j[0]), float(j[1]), float(j[2]), float(j[3]));
            } else {
                sv.SkinIndex = glm::vec4(0, 0, 0, 0);
            }

            if (aWgt) {
                float w[4] {1,0,0,0};
                cgltf_accessor_read_float(aWgt, v, w, 4);
                sv.SkinWeight = glm::vec4(w[0], w[1], w[2], w[3]);
                float sum = sv.SkinWeight.x + sv.SkinWeight.y + sv.SkinWeight.z + sv.SkinWeight.w;
                if (sum > 0.0f) sv.SkinWeight /= sum;
            } else {
                sv.SkinWeight = glm::vec4(1, 0, 0, 0);
            }

            _cpuVertices[v] = sv;
        }

        // Indices
        _indices.clear();
        if (prim->indices) {
            _indices.resize(prim->indices->count);
            for (cgltf_size i = 0; i < prim->indices->count; ++i) {
                _indices[i] = static_cast<std::uint32_t>(cgltf_accessor_read_index(prim->indices, i));
            }
        } else {
            _indices.resize(vertexCount);
            for (cgltf_size i = 0; i < vertexCount; ++i) _indices[i] = static_cast<std::uint32_t>(i);
        }

        return true;
    }

    bool GltfSkinnedModel::BuildSkinnedMesh(cgltf_data const * data) {
        // Find first node that has both mesh and skin
        cgltf_node const * meshNode = nullptr;
        cgltf_skin const * skin = nullptr;

        for (cgltf_size i = 0; i < data->nodes_count; ++i) {
            cgltf_node const * n = &data->nodes[i];
            if (n->mesh && n->skin) {
                meshNode = n;
                skin = n->skin;
                _meshNode = int(i);
                break;
            }
        }
        if (!meshNode || !skin) return false;

        // World matrices for bind pose
        ComputeWorldMatrices();

        if (!BuildSkin(data, skin, meshNode)) return false;
        if (!BuildGeometry(data, meshNode->mesh)) return false;

        // Cache bind world matrices for all nodes (used for robust retargeting space conversions).
        _nodesBindWorld.resize(_nodes.size());
        for (int i = 0; i < int(_nodes.size()); ++i) {
            _nodesBindWorld[i] = _nodes[i].World;
        }

        // Default pose cache
        _poseLocalRot.assign(_skin.JointNodeIndices.size(), glm::quat(1,0,0,0));
        _poseRootDelta = glm::vec3(0.0f);
        _poseHipJoint = -1;
        _poseHipDelta = glm::vec3(0.0f);

        return true;
    }

    bool GltfSkinnedModel::Load(std::filesystem::path const & path) {
        Clear();
        _sourcePath = path.string();

        cgltf_options opt {};
        cgltf_data * data = nullptr;

        if (cgltf_parse_file(&opt, _sourcePath.c_str(), &data) != cgltf_result_success || !data) {
            return false;
        }
        if (cgltf_load_buffers(&opt, data, _sourcePath.c_str()) != cgltf_result_success) {
            cgltf_free(data);
            return false;
        }
        // Optional validation
        (void)cgltf_validate(data);

        BuildNodes(data);
        // Bind pose world matrices from node.LocalBind
        for (auto & n : _nodes) n.Local = n.LocalBind;
        ComputeWorldMatrices();

        if (!BuildSkinnedMesh(data)) {
            cgltf_free(data);
            Clear();
            return false;
        }

        cgltf_free(data);

        EnsureRenderItem();
        _loaded = true;
        // Initialize bone texture for the first draw
        UpdateSkinning();
        return true;
    }

    void GltfSkinnedModel::SetPose(
        std::vector<glm::quat> const & jointLocalRotations,
        glm::vec3 const &              rootTranslationDelta,
        int                            hipJointIndex,
        glm::vec3 const &              hipTranslationDelta) {
        if (!_loaded) return;

        _poseLocalRot = jointLocalRotations;
        if (_poseLocalRot.size() < _skin.JointNodeIndices.size())
            _poseLocalRot.resize(_skin.JointNodeIndices.size(), glm::quat(1,0,0,0));
        _poseRootDelta = rootTranslationDelta;
        _poseHipJoint  = hipJointIndex;
        _poseHipDelta  = hipTranslationDelta;

        // Apply to node locals for joints (keep bind translation/scale)
        for (int j = 0; j < int(_skin.JointNodeIndices.size()); ++j) {
            int nodeIndex = _skin.JointNodeIndices[j];
            auto & node = _nodes[nodeIndex];
            glm::vec3 t = node.BindTranslation;
            if (_skin.JointParent[j] == -1) t += _poseRootDelta;
            if (j == _poseHipJoint)         t += _poseHipDelta;
            glm::quat r = glm::normalize(_poseLocalRot[j]);
            glm::vec3 s = node.BindScale;
            node.Local = TRS(t, r, s);
        }

        // Non-joint nodes: keep bind pose
        for (int i = 0; i < int(_nodes.size()); ++i) {
            if (_skin.NodeToJoint.find(i) == _skin.NodeToJoint.end()) {
                _nodes[i].Local = _nodes[i].LocalBind;
            }
        }
    }

    glm::vec3 GltfSkinnedModel::JointBindWorldPosition(int jointIndex) const {
        if (!_loaded) return glm::vec3(0.0f);
        if (jointIndex < 0 || jointIndex >= int(_skin.JointNodeIndices.size())) return glm::vec3(0.0f);
        int nodeIndex = _skin.JointNodeIndices[jointIndex];
        if (nodeIndex < 0 || nodeIndex >= int(_nodesBindWorld.size())) return glm::vec3(0.0f);
        return glm::vec3(_nodesBindWorld[nodeIndex][3]);
    }

    glm::mat4 GltfSkinnedModel::JointExternalParentBindWorldMatrix(int jointIndex) const {
        if (!_loaded) return glm::mat4(1.0f);
        if (jointIndex < 0 || jointIndex >= int(_skin.JointNodeIndices.size())) return glm::mat4(1.0f);
        int nodeIndex = _skin.JointNodeIndices[jointIndex];
        if (nodeIndex < 0 || nodeIndex >= int(_nodes.size())) return glm::mat4(1.0f);
        int p = _nodes[nodeIndex].Parent;
        if (p < 0) return glm::mat4(1.0f);
        if (p >= int(_nodesBindWorld.size())) return glm::mat4(1.0f);
        return _nodesBindWorld[p];
    }

    glm::quat GltfSkinnedModel::JointExternalParentBindWorldRotation(int jointIndex) const {
        glm::mat4 m = JointExternalParentBindWorldMatrix(jointIndex);
        return WorldRotationFromMat4(m);
    }

    void GltfSkinnedModel::UpdateSkinning() {
        if (!_loaded) return;
        ComputeWorldMatrices();

        // Update joint/bone matrices
        for (int j = 0; j < int(_skin.JointNodeIndices.size()); ++j) {
            int nodeIndex = _skin.JointNodeIndices[j];
            _skin.JointWorldMatrices[j] = _nodes[nodeIndex].World;
            _skin.BoneMatrices[j] = _skin.JointWorldMatrices[j] * _skin.InverseBindMatrices[j];
        }

        // GPU skinning uses a bone texture (three.js-style)
        EnsureBoneTexture();
        UpdateBoneTexture();
    }

    void GltfSkinnedModel::Draw(glm::mat4 const & view, glm::mat4 const & proj, glm::vec4 const & color) {
        if (!_loaded || !_renderItem.has_value() || !_program.has_value()) return;

        // Basic uniforms
        _program->GetUniforms().SetByName("u_View", view);
        _program->GetUniforms().SetByName("u_Projection", proj);
        _program->GetUniforms().SetByName("u_Color", color);

        _program->GetUniforms().SetByName("u_Model", _meshWorld);
        _program->GetUniforms().SetByName("u_BindMatrix", _skin.BindMatrix);
        _program->GetUniforms().SetByName("u_BindMatrixInverse", _skin.BindMatrixInverse);
        // Bind bone texture + uniforms
        if (_boneTexture != 0) {
            glUseProgram(_program->Get());
            if (_uBoneTextureLoc >= 0) glUniform1i(_uBoneTextureLoc, 0);
            if (_uBoneTextureSizeLoc >= 0) glUniform1i(_uBoneTextureSizeLoc, _boneTextureSize);
            glUseProgram(0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _boneTexture);
        }

        _renderItem->Draw({ _program->Use() });

        if (_boneTexture != 0) {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    void GltfSkinnedModel::SkinPositionsCPU(
        std::vector<glm::vec3> & outWorldPositions,
        std::vector<std::uint32_t> const * subsetIndices) const {
        outWorldPositions.clear();
        if (!_loaded) return;

        auto skinOne = [&](SkinnedVertex const & vtx) -> glm::vec3 {
            glm::vec4 bindPos = _skin.BindMatrix * glm::vec4(vtx.Position, 1.0f);
            glm::ivec4 j = glm::ivec4(vtx.SkinIndex + glm::vec4(0.5f));
            glm::vec4 w = vtx.SkinWeight;
            glm::vec4 skinned(0.0f);
            auto const & bm = _skin.BoneMatrices;
            // Robust bounds
            auto mul = [&](int idx) -> glm::mat4 {
                if (idx < 0 || idx >= int(bm.size())) return glm::mat4(1.0f);
                return bm[idx];
            };
            skinned += mul(j.x) * bindPos * w.x;
            skinned += mul(j.y) * bindPos * w.y;
            skinned += mul(j.z) * bindPos * w.z;
            skinned += mul(j.w) * bindPos * w.w;
            glm::vec4 local = _skin.BindMatrixInverse * skinned;
            glm::vec4 world = _meshWorld * local;
            return glm::vec3(world);
        };

        if (!subsetIndices) {
            outWorldPositions.resize(_cpuVertices.size());
            for (size_t i = 0; i < _cpuVertices.size(); ++i) {
                outWorldPositions[i] = skinOne(_cpuVertices[i]);
            }
            return;
        }

        outWorldPositions.reserve(subsetIndices->size());
        for (auto idx : *subsetIndices) {
            if (idx < _cpuVertices.size()) {
                outWorldPositions.push_back(skinOne(_cpuVertices[idx]));
            }
        }
    }

    bool GltfSkinnedModel::RaycastCPU(glm::vec3 const & rayOrigin, glm::vec3 const & rayDir, RaycastHit & outHit) const {
        if (!_loaded || _indices.empty() || _cpuVertices.empty()) return false;

        float bestT = std::numeric_limits<float>::infinity();
        std::uint32_t bestTri = 0;
        glm::vec3 bestPos(0.0f);
        glm::vec3 bestNrm(0.0f, 1.0f, 0.0f);

        // On-demand skinning per triangle: only transform the 3 touched vertices.
        auto skinOne = [&](SkinnedVertex const & vtx) -> glm::vec3 {
            glm::vec4 bindPos = _skin.BindMatrix * glm::vec4(vtx.Position, 1.0f);
            glm::ivec4 j = glm::ivec4(vtx.SkinIndex + glm::vec4(0.5f));
            glm::vec4 w = vtx.SkinWeight;
            glm::vec4 skinned(0.0f);
            auto const & bm = _skin.BoneMatrices;
            auto mul = [&](int idx) -> glm::mat4 {
                if (idx < 0 || idx >= int(bm.size())) return glm::mat4(1.0f);
                return bm[idx];
            };
            skinned += mul(j.x) * bindPos * w.x;
            skinned += mul(j.y) * bindPos * w.y;
            skinned += mul(j.z) * bindPos * w.z;
            skinned += mul(j.w) * bindPos * w.w;
            glm::vec4 local = _skin.BindMatrixInverse * skinned;
            glm::vec4 world = _meshWorld * local;
            return glm::vec3(world);
        };

        glm::vec3 const rd = glm::normalize(rayDir);
        std::uint32_t const triCount = std::uint32_t(_indices.size() / 3);
        for (std::uint32_t t = 0; t < triCount; ++t) {
            std::uint32_t const i0 = _indices[t * 3 + 0];
            std::uint32_t const i1 = _indices[t * 3 + 1];
            std::uint32_t const i2 = _indices[t * 3 + 2];
            if (i0 >= _cpuVertices.size() || i1 >= _cpuVertices.size() || i2 >= _cpuVertices.size()) continue;

            glm::vec3 const v0 = skinOne(_cpuVertices[i0]);
            glm::vec3 const v1 = skinOne(_cpuVertices[i1]);
            glm::vec3 const v2 = skinOne(_cpuVertices[i2]);

            float hitT, hitU, hitV;
            if (!IntersectRayTriangle(rayOrigin, rd, v0, v1, v2, hitT, hitU, hitV)) continue;
            if (hitT < bestT) {
                bestT = hitT;
                bestTri = t;
                bestPos = rayOrigin + rd * hitT;
                bestNrm = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            }
        }

        if (!std::isfinite(bestT)) return false;
        outHit.TriangleIndex = bestTri;
        outHit.T = bestT;
        outHit.Position = bestPos;
        outHit.Normal = bestNrm;
        return true;
    }
} // namespace VCX::Labs::Animation