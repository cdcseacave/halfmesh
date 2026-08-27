/*
* GltfImageCodec.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// OpenCV-backed image codec callbacks for tinygltf.
//
// halfmesh compiles tinygltf with TINYGLTF_NO_STB_IMAGE / TINYGLTF_NO_STB_IMAGE_WRITE
// (see the top-level CMakeLists), so the bundled stb_image encoders/decoders are
// not compiled in at all and the library decodes/encodes every glTF texture with
// the OpenCV imgcodecs it already links.  With those macros set, tinygltf leaves
// TinyGLTF::LoadImageData / WriteImageData null and refuses to parse or serialize
// any image, so EVERY tinygltf::TinyGLTF context that touches images must call
// SetGltfImageCodec() on itself first.
//
// Library-internal header (lives under src/, not include/halfmesh/): no public
// API surface, never installed. Namespace halfmesh::detail.
#pragma once

#if !defined(TINYGLTF_NO_STB_IMAGE) || !defined(TINYGLTF_NO_STB_IMAGE_WRITE)
	#error "halfmesh builds tinygltf with TINYGLTF_NO_STB_IMAGE and TINYGLTF_NO_STB_IMAGE_WRITE; \
these must be defined for every TU that includes tiny_gltf.h (they change the TinyGLTF class \
layout, so a mismatch is an ODR violation). Link halfmesh::halfmesh to inherit them."
#endif

#include <tiny_gltf.h>

#include <string>

namespace halfmesh {
namespace detail {

// tinygltf LoadImageDataFunction. Decodes an encoded blob (PNG/JPEG/...) with
// cv::imdecode and stores it in `image` using tinygltf's own layout: gray /
// RGB / RGBA, 8 or 16 bits per channel, `component` and `bits` mirroring the
// source file. `user_data` is unused (SetGltfImageCodec installs a null one).
bool GltfLoadImageData(tinygltf::Image* image, int image_idx, std::string* err,
                       std::string* warn, int req_width, int req_height,
                       const unsigned char* bytes, int size, void* user_data);

// tinygltf WriteImageDataFunction. Encodes `image` with cv::imencode, picking
// the codec from `filename`'s extension (png/jpg/bmp), then either inlines the
// result as a base64 data URI (embedImages) or writes a sidecar file through
// `fs_cb`. `user_data` is unused.
bool GltfWriteImageData(const std::string* basepath, const std::string* filename,
                        const tinygltf::Image* image, bool embedImages,
                        const tinygltf::FsCallbacks* fs_cb,
                        const tinygltf::URICallbacks* uri_cb, std::string* out_uri,
                        void* user_data);

// Install both callbacks on `ctx`. Must be called before any Load*/Write* on a
// context that may see images -- without it tinygltf fails with
// "No LoadImageData callback specified." / drops the image on write.
void SetGltfImageCodec(tinygltf::TinyGLTF& ctx);

} // namespace detail
} // namespace halfmesh
