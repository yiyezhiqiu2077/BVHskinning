#pragma once

#include <vector>

#include "Engine/app.h"
#include "Labs/4-Animation/CaseBVH.h"
#include "Labs/4-Animation/CaseBVHSkinned.h"
#include "Labs/Common/UI.h"

namespace VCX::Labs::Animation {

    class App : public Engine::IApp {
    private:
        Common::UI             _ui;

        CaseBVH                _caseBVH;
        CaseBVHSkinned         _caseBVHSkinned;

        std::size_t        _caseId = 0;

        // Lab 4 focuses on BVH.
        // We keep the original BVH loader/player and add a BVH->glTF skinned retarget demo.
        std::vector<std::reference_wrapper<Common::ICase>> _cases = { _caseBVH, _caseBVHSkinned };

    public:
        App();
        void OnFrame() override;
    };
}
