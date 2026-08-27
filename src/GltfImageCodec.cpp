/*
* GltfImageCodec.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Implements: the OpenCV-backed image codec declared in GltfImageCodec.h -- and,
// in the same TU, tinygltf itself (https://github.com/syoyo/tinygltf), which is
// header-only and must be expanded in exactly *one* .cpp file.
//
// The two are fused deliberately: GltfWriteImageData needs tinygltf's own
// base64_encode / JoinPath / GetFilePathExtension helpers, and those exist only
// behind TINYGLTF_IMPLEMENTATION -- so the codec cannot live in a TU of its own.
//
// The exception / image-backend macros (TINYGLTF_NOEXCEPTION, JSON_NOEXCEPTION,
// TINYGLTF_NO_STB_IMAGE, TINYGLTF_NO_STB_IMAGE_WRITE) are NOT defined here: the
// image ones change the TinyGLTF class layout, so they must reach every TU that
// includes tiny_gltf.h and are set as target compile definitions instead (see
// the top-level CMakeLists).
//
// Include order matters: tiny_gltf.h's implementation section sits OUTSIDE its
// include guard, so a second #include with TINYGLTF_IMPLEMENTATION still set
// re-emits every definition. Pull the declarations in first (through our own
// header), then expand the implementation exactly once.
#include "GltfImageCodec.h"

#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace halfmesh {
namespace detail {

namespace {

// Accumulate a tinygltf-style diagnostic and report failure.
bool ImageError(std::string* err, int image_idx, const std::string& name, const char* what)
{
	if (err)
		*err += std::string(what) + " for image[" + std::to_string(image_idx) + "] name = \"" + name + "\"\n";
	return false;
}

} // anonymous namespace

bool GltfLoadImageData(tinygltf::Image* image, const int image_idx, std::string* err,
                       std::string* warn, int req_width, int req_height,
                       const unsigned char* bytes, int size, void* user_data)
{
	(void)warn;
	(void)user_data; // halfmesh installs the codec with a null user pointer

	if (bytes == nullptr || size <= 0)
		return ImageError(err, image_idx, image->name, "Empty image data");

	cv::Mat decoded;
	try {
		// imdecode only reads the blob, so wrapping it in a Mat view (which
		// wants a non-const pointer) copies nothing and mutates nothing.
		const cv::Mat encoded(1, size, CV_8UC1, const_cast<unsigned char*>(bytes));
		// IMREAD_UNCHANGED keeps the file's own channel count and bit depth --
		// tinygltf::Image::component/bits are documented to mirror the source,
		// and LoadGLTF's GltfImageToBGR handles 1/3/4 channels either way.
		decoded = cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
	} catch (const std::exception&) {
		// OpenCV reports most decode failures as an empty Mat, but throws on a
		// few (allocation, internal assertion); neither must escape into
		// tinygltf, which does not unwind.
		decoded = cv::Mat();
	}
	if (decoded.empty())
		return ImageError(err, image_idx, image->name, "OpenCV cannot decode image data");
	if (decoded.depth() != CV_8U && decoded.depth() != CV_16U)
		return ImageError(err, image_idx, image->name,
		                  "Unsupported bit depth (glTF images are 8- or 16-bit)");
	if (req_width > 0 && req_width != decoded.cols)
		return ImageError(err, image_idx, image->name, "Image width mismatch");
	if (req_height > 0 && req_height != decoded.rows)
		return ImageError(err, image_idx, image->name, "Image height mismatch");

	// tinygltf stores channels in RGB(A) order; OpenCV decodes them as BGR(A).
	cv::Mat rgb;
	switch (decoded.channels()) {
	case 1: rgb = decoded; break;
	case 3: cv::cvtColor(decoded, rgb, cv::COLOR_BGR2RGB); break;
	case 4: cv::cvtColor(decoded, rgb, cv::COLOR_BGRA2RGBA); break;
	default: return ImageError(err, image_idx, image->name, "Unsupported channel count");
	}

	image->width = rgb.cols;
	image->height = rgb.rows;
	image->component = rgb.channels();
	image->bits = rgb.depth() == CV_16U ? 16 : 8;
	image->pixel_type = rgb.depth() == CV_16U ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
	                                          : TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
	image->as_is = false;
	// imdecode returns a continuous buffer and cvtColor allocates one, so the
	// pixels are a single contiguous run in both branches above.
	const size_t nbytes = rgb.total() * rgb.elemSize();
	image->image.resize(nbytes);
	std::memcpy(image->image.data(), rgb.data, nbytes);
	return true;
}

bool GltfWriteImageData(const std::string* basepath, const std::string* filename,
                        const tinygltf::Image* image, bool embedImages,
                        const tinygltf::FsCallbacks* fs_cb,
                        const tinygltf::URICallbacks* uri_cb, std::string* out_uri,
                        void* user_data)
{
	(void)user_data;

	// tinygltf invokes the writer for image slots with no pixels too; report the
	// filename as the URI and write nothing (same as its stb-backed default).
	if (image->image.empty()) {
		*out_uri = *filename;
		return true;
	}

	// tinygltf derives the filename extension from the image's mimeType, so the
	// extension is what actually selects the encoder here.
	const std::string ext = tinygltf::GetFilePathExtension(*filename);
	const char* header = nullptr; // data-URI prefix for the embedded case
	const char* cvExt = nullptr; // extension cv::imencode picks its codec from
	std::vector<int> params;
	if (ext == "png") {
		header = "data:image/png;base64,";
		cvExt = ".png";
	} else if (ext == "jpg" || ext == "jpeg") {
		header = "data:image/jpeg;base64,";
		cvExt = ".jpg";
		// Match the quality stb_image_write hardcoded, so switching codec does
		// not silently degrade exports. OpenCV's own default would be 95.
		// PNG deliberately passes no params: OpenCV's default beats every
		// explicit IMWRITE_PNG_COMPRESSION level on both size and time.
		params = {cv::IMWRITE_JPEG_QUALITY, 100};
	} else if (ext == "bmp") {
		header = "data:image/bmp;base64,";
		cvExt = ".bmp";
	} else {
		return false; // no encoder for this extension
	}

	std::vector<unsigned char> data;
	if (image->as_is) {
		// Already-encoded blob (a delayed-decode loader kept it verbatim): the
		// bytes are the file, pass them through untouched.
		data = image->image;
	} else {
		if (image->width <= 0 || image->height <= 0 || image->component <= 0)
			return false;
		if (image->bits != 8 && image->bits != 16)
			return false;
		const size_t texels = static_cast<size_t>(image->width) * static_cast<size_t>(image->height);
		const size_t expected = texels * static_cast<size_t>(image->component * (image->bits / 8));
		if (image->image.size() < expected)
			return false;
		const cv::Mat rgb(image->height, image->width,
		                  CV_MAKETYPE(image->bits == 16 ? CV_16U : CV_8U, image->component),
		                  const_cast<unsigned char*>(image->image.data()));
		cv::Mat bgr;
		switch (image->component) {
		case 1: bgr = rgb; break;
		case 3: cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR); break;
		case 4: cv::cvtColor(rgb, bgr, cv::COLOR_RGBA2BGRA); break;
		default: return false;
		}
		try {
			if (!cv::imencode(cvExt, bgr, data, params) || data.empty())
				return false;
		} catch (const std::exception&) {
			// e.g. JPEG asked for 16-bit or 4-channel input: a hard OpenCV
			// error, which must surface as a write failure, not an unwind
			// through tinygltf.
			return false;
		}
	}

	if (embedImages) {
		*out_uri = header + tinygltf::base64_encode(data.data(), static_cast<unsigned int>(data.size()));
		return true;
	}

	// Sidecar file: go through tinygltf's filesystem callbacks so a caller that
	// overrode them keeps control of where the bytes land.
	if (fs_cb == nullptr || fs_cb->WriteWholeFile == nullptr)
		return false;
	std::string writeError;
	if (!fs_cb->WriteWholeFile(&writeError, tinygltf::JoinPath(*basepath, *filename), data,
	                           fs_cb->user_data))
		return false;
	if (uri_cb != nullptr && uri_cb->encode)
		return uri_cb->encode(*filename, "image", out_uri, uri_cb->user_data);
	*out_uri = *filename;
	return true;
}

void SetGltfImageCodec(tinygltf::TinyGLTF& ctx)
{
	ctx.SetImageLoader(&GltfLoadImageData, nullptr);
	ctx.SetImageWriter(&GltfWriteImageData, nullptr);
}

} // namespace detail
} // namespace halfmesh
