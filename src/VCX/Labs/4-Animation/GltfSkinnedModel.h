#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Engine/GL/RenderItem.h"
#include "Engine/GL/Program.h"

// cgltf include path varies across package managers.
// xmake's `cgltf` package usually exposes either <cgltf.h> or <cgltf/cgltf.h>.
#if __has_include(<cgltf/cgltf.h>)
#include <cgltf/cgltf.h>
#elif __has_include(<cgltf.h>)
#include <cgltf.h>
#else
#error "cgltf headers not found. Make sure xmake has installed the cgltf package."
#endif

namespace VCX::Labs::Animation {
    struct SkinnedVertex {
        glm::vec3 Position {};
        glm::vec3 Normal {};
        glm::vec2 TexCoord {};
        glm::vec4 SkinIndex {};  // floats
        glm::vec4 SkinWeight {};
    };

    struct GltfJointInfo {
        std::string Name;
        int         ParentJoint {-1};          // parent joint index in skin order
        int         NodeIndex   {-1};          // corresponding glTF node index

        // Bind-pose local transform (from parent joint space to this joint space).
        // Translation defines bone length; rotations are used for BVH retargeting.
        glm::vec3   BindLocalTranslation {0.0f};
        glm::quat   BindLocalRotation    {1, 0, 0, 0};

        // Optional debug info (bind-pose world rotation).
        glm::quat   BindWorldRotation    {1, 0, 0, 0};
    };

    struct GltfSkinInstance {
        // In glTF skin order
        std::vector<int>       JointNodeIndices;
        std::vector<int>       JointParent; // parent joint index in skin order (-1 for root)
        std::vector<glm::mat4> InverseBindMatrices;

        // Updated every frame
        std::vector<glm::mat4> JointWorldMatrices;
        std::vector<glm::mat4> BoneMatrices; // jointWorld * inverseBind

        // Helper mapping nodeIndex -> jointIndex (in skin order)
        std::unordered_map<int, int> NodeToJoint;

        // Bind matrices (mesh.bindMatrix/bindMatrixInverse)
        glm::mat4 BindMatrix        {1.0f};
        glm::mat4 BindMatrixInverse {1.0f};
    };

    // Minimal glTF 2.0 skinned model loader for Lab4.
    // - Loads a single skinned mesh node (first node that references both a mesh + skin)
    // - Builds GPU buffers for GPU skinning
    // - Exposes bind-pose + skeleton info for BVH retargeting
    // - Supports on-demand CPU skinning for precise geometry queries (picking/collision/baking)
    class GltfSkinnedModel {
    public:
        bool Load(std::filesystem::path const & path);
        void Clear();
        bool IsLoaded() const { return _loaded; }

        // Pose control (joint-local rotations in skin order)
        // - Non-root joints keep bind translation (bone lengths).
        // - RootTranslationDelta is applied additively on the skeleton root (useful for BVH root motion).
        // - Optionally, HipTranslationDelta can be applied additively on a chosen "hip" joint.
        void SetPose(
            std::vector<glm::quat> const & jointLocalRotations,
            glm::vec3 const &              rootTranslationDelta,
            int                            hipJointIndex,
            glm::vec3 const &              hipTranslationDelta);

        // Convenience overload (no hip translation).
        void SetPose(
            std::vector<glm::quat> const & jointLocalRotations,
            glm::vec3 const &              rootTranslationDelta) {
            SetPose(jointLocalRotations, rootTranslationDelta, -1, glm::vec3(0.0f));
        }

        // Recompute joint matrices + upload to GPU.
        void UpdateSkinning();

        // Draw (flat color).
        void Draw(glm::mat4 const & view, glm::mat4 const & proj, glm::vec4 const & color);

        // Accessors for retargeting
        std::vector<GltfJointInfo> const & Joints() const { return _jointInfos; }
        GltfSkinInstance const &          Skin() const { return _skin; }
        glm::mat4 const &                 MeshWorldMatrix() const { return _meshWorld; }

        // Bind-pose helpers (world space)
        glm::vec3 JointBindWorldPosition(int jointIndex) const;
        glm::mat4 JointExternalParentBindWorldMatrix(int jointIndex) const;
        glm::quat JointExternalParentBindWorldRotation(int jointIndex) const;

        // CPU skinning (world positions) for either all vertices or a subset.
        // This is intentionally "local/on-demand" like Three.js's `SkinnedMesh.boneTransform`.
        void SkinPositionsCPU(
            std::vector<glm::vec3> & outWorldPositions,
            std::vector<std::uint32_t> const * subsetIndices = nullptr) const;

        struct RaycastHit {
            float       T {0.0f};
            glm::vec3   Position {0.0f};
            glm::vec3   Normal {0.0f, 1.0f, 0.0f};
            std::uint32_t TriangleIndex {0}; // index into the triangle list (0..numTris-1)
        };

        // CPU raycast against the deformed mesh, skinning only the touched vertices per triangle.
        // This mirrors the spirit of three.js SkinnedMesh raycast path (boneTransform on-demand).
        bool RaycastCPU(glm::vec3 const & rayOrigin, glm::vec3 const & rayDir, RaycastHit & outHit) const;

        // Debug
        std::string const & SourcePath() const { return _sourcePath; }

    private:
        struct Node {
            std::string Name;
            int         Parent {-1};
            std::vector<int> Children;
            glm::mat4   LocalBind {1.0f};
            glm::vec3   BindTranslation {0.0f};
            glm::quat   BindRotation {1, 0, 0, 0};
            glm::vec3   BindScale {1.0f};

            // Mutable pose
            glm::mat4   Local {1.0f};
            glm::mat4   World {1.0f};
        };

        bool _loaded {false};
        std::string _sourcePath;

        // Parsed scene graph subset (only what's needed for skinning)
        std::vector<Node> _nodes;

        // Cached bind-pose node world matrices (same indexing as _nodes).
        // Used for robust space conversions (root motion / hip translation / offsets).
        std::vector<glm::mat4> _nodesBindWorld;

        // Mesh node
        int       _meshNode {-1};
        glm::mat4 _meshWorld {1.0f};

        // Skin (only one)
        GltfSkinInstance          _skin;
        std::vector<GltfJointInfo> _jointInfos;

        // Geometry (CPU copy + GPU render item)
        std::vector<SkinnedVertex>  _cpuVertices;
        std::vector<std::uint32_t>  _indices;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _renderItem;

        // Shader program (GPU skinning)
        std::optional<Engine::GL::UniqueProgram> _program;

        // Bone texture (three.js-style GPU skinning without large uniform arrays)
        unsigned int _boneTexture {0};      // GL texture id
        int          _boneTextureSize {0};  // texture width/height (square)
        std::vector<float> _boneTexturePixels; // RGBA32F pixels (size*size*4)

        // Uniform locations
        int _uBoneTextureLoc {-1};      // location of u_BoneTexture
        int _uBoneTextureSizeLoc {-1};  // location of u_BoneTextureSize

        // Pose cache
        std::vector<glm::quat> _poseLocalRot;
        glm::vec3             _poseRootDelta {0};
        int                   _poseHipJoint {-1};
        glm::vec3             _poseHipDelta {0};


    private:
        static glm::mat4 MakeLocalMatrix(cgltf_node const * n);
        void             BuildNodes(cgltf_data const * data);
        void             ComputeWorldMatrices();
        bool             BuildSkinnedMesh(cgltf_data const * data);
        bool             BuildSkin(cgltf_data const * data, cgltf_skin const * skin, cgltf_node const * meshNode);
        bool             BuildGeometry(cgltf_data const * data, cgltf_mesh const * mesh);
        void             EnsureShaders();
        void             EnsureRenderItem();
        void             EnsureBoneTexture();
        void             UpdateBoneTexture();
    };
} // namespace VCX::Labs::Animation
