# Overlay port: tinygltf 3.0.0#1
#
# WHY THIS EXISTS
# The tinygltf port at our pinned `builtin-baseline` carries a stale SHA512 for
# the GitHub release tarball. The upstream tag was re-cut, so GitHub now serves
# different bytes and every fetch fails the integrity check:
#
#   error: download from https://github.com/syoyo/tinygltf/archive/v3.0.0.tar.gz
#          had an unexpected hash
#
# That is fatal on a cold cache — which is every CI runner — on all three
# platforms. Machines with a warm vcpkg download cache never re-verify, which is
# why local builds kept working while CI never did.
#
# Upstream vcpkg has already corrected this in tinygltf 3.0.0#1; this file is a
# verbatim copy of that fixed port. We cannot reach it through `overrides`
# because port-version 1 does not exist in the version database at our baseline,
# and bumping the baseline would re-resolve every other dependency.
#
# DELETE THIS OVERLAY (and vcpkg-configuration.json, if it holds nothing else)
# once `builtin-baseline` in vcpkg.json advances to a vcpkg commit that includes
# tinygltf 3.0.0#1.

# Header-only library
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO syoyo/tinygltf
    REF "v${VERSION}"
    SHA512 b9a689f25284206969f31ba382abd06cd4ed69ee6a118af3484aaa17b68aebefddc97a4e722e3bde8fa39b737677cb09cbd3d3e6f2a573ac4ba1fb718478b79a
    HEAD_REF master
)

# Put the licence file where vcpkg expects it
# Copy the tinygltf header files and fix the path to json
vcpkg_replace_string("${SOURCE_PATH}/tiny_gltf.h" "#include \"json.hpp\"" "#include <nlohmann/json.hpp>")
file(INSTALL
        "${SOURCE_PATH}/tiny_gltf.h"
        "${SOURCE_PATH}/tiny_gltf_v3.h"
        "${SOURCE_PATH}/tinygltf_json.h"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include"
)
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
