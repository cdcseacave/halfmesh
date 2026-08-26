/*
* MeshIOTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Mesh I/O tests: PLY round-trip (Mesh::Load / Mesh::SavePLY), glTF
// save + load round-trip (Mesh::SaveGLTF / Mesh::LoadGLTF), and mesh.ply-based
// TriangleKdTree tests.

#include <gtest/gtest.h>

#include <halfmesh/Mesh.h>
#include <halfmesh/TriangleKDTree.h>
#include <halfmesh/Util/Geometry.h>
#include <halfmesh/Util/Assert.h>

#include <tiny_gltf.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers shared by KdTree tests
// ---------------------------------------------------------------------------

inline unsigned RandomRange(unsigned start, unsigned end)
{
	return start + static_cast<unsigned>(std::round((end - start - 1) * (static_cast<double>(std::rand()) / RAND_MAX)));
}

inline Eigen::Vector3f RandomBarycenter()
{
	Eigen::Vector3f bary = Eigen::Vector3f::Random();
	if (bary.x() < 0)
		bary.x() *= -1;
	if (bary.y() < 0)
		bary.y() *= -1;
	while (bary.x() + bary.y() > 1) {
		bary.x() *= 0.5f;
		bary.y() *= 0.5f;
	}
	bary.z() = 1.f - (bary.x() + bary.y());
	return bary;
}

// Path to the test mesh (committed to data/ in the repo root).
std::string TestMeshPath()
{
	// __FILE__ is tests/MeshIOTest.cpp → go one level up.
	return (std::filesystem::path(__FILE__).parent_path()
	        / "data" / "mesh.ply")
	    .string();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// PLY round-trip
// ---------------------------------------------------------------------------
TEST(MeshIoTest, PLYRoundTrip)
{
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()))
	    << "Load failed for: " << TestMeshPath();
	ASSERT_FALSE(mesh.Empty());

	const std::size_t originalVertices = mesh.vertices.size();
	const std::size_t originalFaces = mesh.faces.size();

	// Save to a temp file
	const std::string tmp = (std::filesystem::temp_directory_path() / "halfmesh_roundtrip.ply").string();
	ASSERT_TRUE(mesh.SavePLY(tmp, /*binary=*/true));

	// Reload
	halfmesh::Mesh mesh2;
	ASSERT_TRUE(mesh2.Load(tmp));
	ASSERT_FALSE(mesh2.Empty());

	// Topology must be identical after round-trip
	EXPECT_EQ(mesh2.vertices.size(), originalVertices);
	EXPECT_EQ(mesh2.faces.size(), originalFaces);

	// Vertex positions must match within float precision
	for (std::size_t i = 0; i < originalVertices; ++i) {
		EXPECT_NEAR((mesh2.vertices[i] - mesh.vertices[i]).norm(), 0.f, 1e-5f)
		    << "vertex " << i << " mismatch";
	}
}

// ---------------------------------------------------------------------------
// glTF save smoke-test (no texture)
// ---------------------------------------------------------------------------
TEST(MeshIoTest, GLTFSaveSmoke)
{
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()));
	ASSERT_FALSE(mesh.Empty());

	const std::string tmpGlb = (std::filesystem::temp_directory_path() / "halfmesh_smoke.glb").string();
	EXPECT_TRUE(mesh.Save(tmpGlb, /*binary=*/true));
	EXPECT_TRUE(std::filesystem::exists(tmpGlb));
	EXPECT_GT(std::filesystem::file_size(tmpGlb), 0u);
}

// ---------------------------------------------------------------------------
// Save dispatch — unknown extension returns false
// ---------------------------------------------------------------------------
TEST(MeshIoTest, SaveUnknownExtension)
{
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()));
	EXPECT_FALSE(mesh.Save("/tmp/halfmesh_test.xyz"));
}

// ---------------------------------------------------------------------------
// mesh.ply-based TriangleKdTree test — NearestPoint on surface
// TriangleKdTree surface iteration tests
// ---------------------------------------------------------------------------
TEST(MeshKdTreeOnRealMesh, NearestPointOnSurface)
{
	std::srand(42);
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()));
	ASSERT_FALSE(mesh.Empty());

	halfmesh::TriangleKdTree tree(mesh);

	const unsigned itersSurface = 100;
	for (unsigned iter = 0; iter < itersSurface; ++iter) {
		const unsigned idxFace = RandomRange(0, static_cast<unsigned>(mesh.faces.size()));
		const halfmesh::Mesh::Face& face = mesh.faces[idxFace];
		const Eigen::Vector3f bary = RandomBarycenter();
		const Eigen::Vector3f X =
		    mesh.vertices[face.x()] * bary.x() + mesh.vertices[face.y()] * bary.y() + mesh.vertices[face.z()] * bary.z();
		const halfmesh::TriangleKdTree::NearestNeighbor nn = tree.NearestPoint(X);
		// A point sampled on a face should map back to that face with zero dist
		EXPECT_EQ(nn.idxFace, idxFace)
		    << "iter=" << iter << " expected face " << idxFace
		    << " got " << nn.idxFace;
		EXPECT_NEAR(nn.dist, 0.f, 1e-4f) << "iter=" << iter;
	}
}

// ---------------------------------------------------------------------------
// mesh.ply-based TriangleKdTree test — NearestPoint brute-force comparison
// TriangleKdTree volume iteration tests
// ---------------------------------------------------------------------------
TEST(MeshKdTreeOnRealMesh, NearestPointVsBruteForce)
{
	std::srand(42);
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()));
	ASSERT_FALSE(mesh.Empty());

	halfmesh::TriangleKdTree tree(mesh);

	const Eigen::Vector3f hsize = tree.GetAABBox().sizes() / 2;
	const Eigen::Vector3f center = tree.GetAABBox().center();

	const unsigned itersVolume = 50;
	for (unsigned iter = 0; iter < itersVolume; ++iter) {
		const Eigen::Vector3f X = center + hsize.cwiseProduct(Eigen::Vector3f::Random());
		const halfmesh::TriangleKdTree::NearestNeighbor nn = tree.NearestPoint(X);

		// Brute-force
		halfmesh::Mesh::FIndex bestIdx = math::NO_ID;
		Eigen::Vector3f bestNearest;
		halfmesh::Mesh::Type bestDist = std::numeric_limits<halfmesh::Mesh::Type>::max();
		FOREACHIDX (halfmesh::Mesh::FIndex, iF, mesh.faces) {
			const halfmesh::Mesh::Face& face = mesh.faces[iF];
			Eigen::Vector3f nearest;
			const halfmesh::Mesh::Type d = math::DistanceBetweenTriangleAndPoint(
			    mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]],
			    X, &nearest);
			if (bestDist > d) {
				bestDist = d;
				bestNearest = nearest;
				bestIdx = iF;
			}
		}
		EXPECT_NEAR(nn.dist, bestDist, 1e-4f) << "iter=" << iter;
		EXPECT_NEAR((nn.nearest - bestNearest).norm(), 0.f, 1e-4f) << "iter=" << iter;
	}
}

// ---------------------------------------------------------------------------
// GLB TEXCOORD_0 regression: UV-only mesh (no texture image) must still
// emit TEXCOORD_0 in the saved GLB.
// ---------------------------------------------------------------------------
TEST(MeshIoTest, GLBUVOnlyHasTexcoord0)
{
	// Build a minimal two-triangle mesh with per-face-corner UV coordinates.
	// This simulates what GenerateAtlas produces: faceTexcoords filled with
	// normalized [0,1] UVs but no texturesDiffuse image.
	halfmesh::Mesh mesh;
	// Four vertices forming a unit square
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 1.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	// Two triangles
	mesh.faces = {
	    halfmesh::Mesh::Face(0, 1, 2),
	    halfmesh::Mesh::Face(0, 2, 3),
	};
	// Per-face-corner UV coords (atlas-style, already normalized [0,1])
	// 2 faces × 3 corners = 6 UVs
	mesh.faceTexcoords = {
	    {0.f, 0.f},
	    {1.f, 0.f},
	    {1.f, 1.f}, // face 0
	    {0.f, 0.f},
	    {1.f, 1.f},
	    {0.f, 1.f}, // face 1
	};
	// No texturesDiffuse — UV-only path
	ASSERT_TRUE(mesh.HasTextureCoordinates());
	ASSERT_FALSE(mesh.HasTexture());

	const std::string tmpGlb = (std::filesystem::temp_directory_path() / "halfmesh_uv_only.glb").string();
	ASSERT_TRUE(mesh.Save(tmpGlb, /*binary=*/true));
	ASSERT_TRUE(std::filesystem::exists(tmpGlb));

	// Parse back with tinygltf and verify TEXCOORD_0 is present
	tinygltf::TinyGLTF loader;
	tinygltf::Model model;
	std::string err, warn;
	ASSERT_TRUE(loader.LoadBinaryFromFile(&model, &err, &warn, tmpGlb))
	    << "tinygltf load error: " << err;

	ASSERT_FALSE(model.meshes.empty());
	ASSERT_FALSE(model.meshes[0].primitives.empty());
	const tinygltf::Primitive& prim = model.meshes[0].primitives[0];
	EXPECT_TRUE(prim.attributes.count("TEXCOORD_0") > 0)
	    << "UV-only GLB must have TEXCOORD_0 attribute";

	// Verify accessor count matches vertex count (per-vertex layout)
	if (prim.attributes.count("TEXCOORD_0")) {
		const int accIdx = prim.attributes.at("TEXCOORD_0");
		ASSERT_GE(accIdx, 0);
		ASSERT_LT(accIdx, static_cast<int>(model.accessors.size()));
		const size_t uvCount = model.accessors[accIdx].count;
		const size_t posCount = model.accessors[prim.attributes.at("POSITION")].count;
		EXPECT_EQ(uvCount, posCount)
		    << "TEXCOORD_0 accessor count must match POSITION accessor count";
		EXPECT_GT(uvCount, 0u);
	}
}

// ---------------------------------------------------------------------------
// Confirm textured path still emits texture + UVs (regression guard).
// Uses a minimal mesh with a 2×2 texture image.
// ---------------------------------------------------------------------------
TEST(MeshIoTest, GLBTexturedPathPreserved)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2)};
	// Pixel-space UVs (absolute, matching a 2×2 texture)
	mesh.faceTexcoords = {
	    {0.f, 0.f},
	    {2.f, 0.f},
	    {0.f, 2.f},
	};
	// Add a minimal 2×2 diffuse texture image (BGR, 8-bit)
	cv::Mat tex(2, 2, CV_8UC3, cv::Scalar(200, 100, 50));
	mesh.texturesDiffuse.emplace_back(tex);
	ASSERT_TRUE(mesh.HasTexture());

	const std::string tmpGlb = (std::filesystem::temp_directory_path() / "halfmesh_textured.glb").string();
	ASSERT_TRUE(mesh.Save(tmpGlb, /*binary=*/true));
	ASSERT_TRUE(std::filesystem::exists(tmpGlb));

	tinygltf::TinyGLTF loader;
	tinygltf::Model model;
	std::string err, warn;
	ASSERT_TRUE(loader.LoadBinaryFromFile(&model, &err, &warn, tmpGlb))
	    << "tinygltf load error: " << err;

	ASSERT_FALSE(model.meshes.empty());
	ASSERT_FALSE(model.meshes[0].primitives.empty());
	const tinygltf::Primitive& prim = model.meshes[0].primitives[0];
	EXPECT_TRUE(prim.attributes.count("TEXCOORD_0") > 0)
	    << "Textured GLB must have TEXCOORD_0 attribute";
	EXPECT_FALSE(model.images.empty())
	    << "Textured GLB must have an embedded texture image";
	EXPECT_FALSE(model.materials.empty());
	if (!model.materials.empty()) {
		EXPECT_GE(model.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0)
		    << "Textured GLB material must reference a texture";
	}
}

// ---------------------------------------------------------------------------
// SaveGLTF's imageFormat / embedImages parameters.
//
// tinygltf chooses its encoder from the image filename extension, which it
// derives from the glTF mimeType -- so asserting on the emitted URI is what
// actually proves the format reached the encoder.  Saved as .gltf (not .glb)
// so the JSON, and with it the URI, is readable straight out of the file.
// ---------------------------------------------------------------------------
namespace {
// Two disjoint triangles, one per texture blob.
halfmesh::Mesh TwoBlobTexturedMesh()
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	    halfmesh::Mesh::Vertex(2.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(3.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(2.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2), halfmesh::Mesh::Face(3, 4, 5)};
	mesh.faceTexcoords = {
	    {0.f, 0.f},
	    {2.f, 0.f},
	    {0.f, 2.f},
	    {0.f, 0.f},
	    {2.f, 0.f},
	    {0.f, 2.f},
	};
	mesh.faceTexblobs = {0, 1}; // face-indexed on input
	mesh.texturesDiffuse.emplace_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(200, 100, 50)));
	mesh.texturesDiffuse.emplace_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(10, 220, 30)));
	return mesh;
}

std::string ReadWholeFile(const std::string& path)
{
	std::ifstream f(path, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
} // anonymous namespace

TEST(MeshIoTest, SaveGLTFEmbeddedImageFormat)
{
	halfmesh::Mesh mesh = TwoBlobTexturedMesh();
	ASSERT_TRUE(mesh.HasTexture());

	const std::string jpgPath =
	    (std::filesystem::temp_directory_path() / "halfmesh_fmt_jpg.gltf").string();
	ASSERT_TRUE(mesh.SaveGLTF(jpgPath, /*binary=*/false, halfmesh::Mesh::ImageFormat::JPG,
	                          /*embedImages=*/true));
	const std::string jpgJson = ReadWholeFile(jpgPath);
	EXPECT_NE(jpgJson.find("data:image/jpeg;base64,"), std::string::npos)
	    << "JPG + embedImages must inline a JPEG data URI";
	EXPECT_EQ(jpgJson.find("data:image/png;base64,"), std::string::npos);

	const std::string pngPath =
	    (std::filesystem::temp_directory_path() / "halfmesh_fmt_png.gltf").string();
	ASSERT_TRUE(mesh.SaveGLTF(pngPath, /*binary=*/false, halfmesh::Mesh::ImageFormat::PNG,
	                          /*embedImages=*/true));
	const std::string pngJson = ReadWholeFile(pngPath);
	EXPECT_NE(pngJson.find("data:image/png;base64,"), std::string::npos)
	    << "PNG + embedImages must inline a PNG data URI";
	EXPECT_EQ(pngJson.find("data:image/jpeg;base64,"), std::string::npos);

	// The default still embeds JPEG, so existing callers are unaffected.
	const std::string defPath =
	    (std::filesystem::temp_directory_path() / "halfmesh_fmt_default.gltf").string();
	ASSERT_TRUE(mesh.SaveGLTF(defPath, /*binary=*/false));
	EXPECT_NE(ReadWholeFile(defPath).find("data:image/jpeg;base64,"), std::string::npos);
}

// Each blob must land in its OWN sidecar: the images are named per blob, and a
// shared name would have every blob overwrite, and then reference, one file.
TEST(MeshIoTest, SaveGLTFExternalImagesArePerBlob)
{
	halfmesh::Mesh mesh = TwoBlobTexturedMesh();
	ASSERT_TRUE(mesh.HasTexture());

	const std::filesystem::path dir =
	    std::filesystem::temp_directory_path() / "halfmesh_gltf_external";
	std::filesystem::remove_all(dir);
	std::filesystem::create_directories(dir);
	const std::string outPath = (dir / "scene.gltf").string();

	ASSERT_TRUE(mesh.SaveGLTF(outPath, /*binary=*/false, halfmesh::Mesh::ImageFormat::PNG,
	                          /*embedImages=*/false));

	const std::filesystem::path tex0 = dir / "scene_diffuse00.png";
	const std::filesystem::path tex1 = dir / "scene_diffuse01.png";
	EXPECT_TRUE(std::filesystem::exists(tex0)) << "missing sidecar: " << tex0;
	EXPECT_TRUE(std::filesystem::exists(tex1)) << "missing sidecar: " << tex1;
	EXPECT_GT(std::filesystem::file_size(tex0), 0u);

	// Nothing inlined, and both sidecars referenced.
	const std::string json = ReadWholeFile(outPath);
	EXPECT_EQ(json.find("data:image/"), std::string::npos)
	    << "embedImages=false must not inline any image";
	EXPECT_NE(json.find("scene_diffuse00.png"), std::string::npos);
	EXPECT_NE(json.find("scene_diffuse01.png"), std::string::npos);

	std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// glTF geometry round-trip: Save(.glb) -> Load(.glb).
//
// SaveGLTF bakes a z-up->y-up node matrix into the scene, so the reloaded mesh
// is a rigid 90 deg rotation of the original.  We therefore assert rotation
// invariants: exact face count, preserved surface area, and a matching multiset
// of AABB extents (the rotation only permutes/sign-flips the axes).
// ---------------------------------------------------------------------------
namespace {
Eigen::Vector3f SortedExtents(const halfmesh::Mesh& mesh)
{
	Eigen::Vector3f e = mesh.ComputeAABBox().sizes();
	std::sort(e.data(), e.data() + 3);
	return e;
}
} // anonymous namespace

TEST(MeshIoTest, GLTFGeometryRoundTrip)
{
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()));
	ASSERT_FALSE(mesh.Empty());

	const std::size_t originalFaces = mesh.faces.size();
	const halfmesh::real originalArea = mesh.ComputeArea();
	const Eigen::Vector3f originalExt = SortedExtents(mesh);

	const std::string tmpGlb = (std::filesystem::temp_directory_path() / "halfmesh_roundtrip.glb").string();
	ASSERT_TRUE(mesh.Save(tmpGlb, /*binary=*/true));

	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.Load(tmpGlb)) << "LoadGLTF failed for: " << tmpGlb;
	ASSERT_FALSE(reloaded.Empty());

	// Face count is exact; vertices may grow if SaveGLTF split UV seams.
	EXPECT_EQ(reloaded.faces.size(), originalFaces);
	EXPECT_GE(reloaded.vertices.size(), mesh.vertices.size());

	// Surface area is preserved under the rigid transform.
	EXPECT_NEAR(reloaded.ComputeArea(), originalArea, originalArea * 1e-3);

	// Sorted AABB extents match (the y-up rotation only permutes the axes).
	const Eigen::Vector3f reloadedExt = SortedExtents(reloaded);
	EXPECT_NEAR((reloadedExt - originalExt).norm(), 0.f, originalExt.norm() * 1e-3f);
}

// ---------------------------------------------------------------------------
// glTF UV-only round-trip: a mesh with normalized [0,1] UVs but no texture
// image (atlas/unwrap output) must reload with its UVs intact.
// ---------------------------------------------------------------------------
TEST(MeshIoTest, GLTFUVOnlyRoundTrip)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 1.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2), halfmesh::Mesh::Face(0, 2, 3)};
	mesh.faceTexcoords = {
	    {0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, // face 0
	    {0.f, 0.f},
	    {1.f, 1.f},
	    {0.f, 1.f}, // face 1
	};
	ASSERT_TRUE(mesh.HasTextureCoordinates());
	ASSERT_FALSE(mesh.HasTexture());

	// Exercise the ASCII (.gltf) path too.
	const std::string tmp = (std::filesystem::temp_directory_path() / "halfmesh_uv_only_rt.gltf").string();
	ASSERT_TRUE(mesh.Save(tmp, /*binary=*/false));

	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.Load(tmp));
	EXPECT_EQ(reloaded.faces.size(), mesh.faces.size());
	ASSERT_TRUE(reloaded.HasTextureCoordinates());
	EXPECT_TRUE(reloaded.texturesDiffuse.empty());
	ASSERT_EQ(reloaded.faceTexcoords.size(), mesh.faceTexcoords.size());
	// UV-only coords stay normalized [0,1]; every corner UV must reappear.
	for (const halfmesh::Mesh::TexCoord& uv : reloaded.faceTexcoords) {
		EXPECT_GE(uv.x(), -1e-4f);
		EXPECT_LE(uv.x(), 1.f + 1e-4f);
		EXPECT_GE(uv.y(), -1e-4f);
		EXPECT_LE(uv.y(), 1.f + 1e-4f);
	}
}

// ---------------------------------------------------------------------------
// glTF textured round-trip: image + absolute-pixel UVs must survive
// Save(.glb) -> Load(.glb) (UVs converted back from glTF-normalized).
// ---------------------------------------------------------------------------
TEST(MeshIoTest, GLTFTexturedRoundTrip)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2)};
	// Absolute pixel-space UVs matching a 4x4 texture.
	mesh.faceTexcoords = {{0.f, 0.f}, {3.f, 0.f}, {0.f, 3.f}};
	cv::Mat tex(4, 4, CV_8UC3, cv::Scalar(200, 100, 50));
	mesh.texturesDiffuse.emplace_back(tex);
	ASSERT_TRUE(mesh.HasTexture());
	const std::vector<halfmesh::Mesh::TexCoord> expectedUv = mesh.faceTexcoords;

	const std::string tmpGlb = (std::filesystem::temp_directory_path() / "halfmesh_textured_rt.glb").string();
	ASSERT_TRUE(mesh.Save(tmpGlb, /*binary=*/true));

	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.Load(tmpGlb));
	EXPECT_EQ(reloaded.faces.size(), 1u);
	ASSERT_TRUE(reloaded.HasTexture());
	ASSERT_FALSE(reloaded.texturesDiffuse.empty());
	EXPECT_EQ(reloaded.texturesDiffuse[0].cols, 4);
	EXPECT_EQ(reloaded.texturesDiffuse[0].rows, 4);

	// Absolute-pixel UVs reconstructed from the glTF-normalized coords.
	ASSERT_EQ(reloaded.faceTexcoords.size(), expectedUv.size());
	for (std::size_t i = 0; i < expectedUv.size(); ++i)
		EXPECT_NEAR((reloaded.faceTexcoords[i] - expectedUv[i]).norm(), 0.f, 1e-3f)
		    << "UV corner " << i << " mismatch";
}

// ---------------------------------------------------------------------------
// glTF load of a corrupt/non-glTF file must fail gracefully (no crash).
// ---------------------------------------------------------------------------
TEST(MeshIoTest, GLTFLoadGarbageFails)
{
	const std::string tmp = (std::filesystem::temp_directory_path() / "halfmesh_not_a.glb").string();
	{
		std::ofstream f(tmp, std::ios::binary);
		f << "this is definitely not a glTF binary";
	}
	halfmesh::Mesh mesh;
	EXPECT_FALSE(mesh.Load(tmp));
	EXPECT_TRUE(mesh.Empty());
}

// ---------------------------------------------------------------------------
// mesh.ply-based TriangleKdTree test — IntersectedPoint
// TriangleKdTree ray intersection tests
// ---------------------------------------------------------------------------
TEST(MeshKdTreeOnRealMesh, IntersectedPoint)
{
	std::srand(42);
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.Load(TestMeshPath()));
	ASSERT_FALSE(mesh.Empty());

	halfmesh::TriangleKdTree tree(mesh);

	const float minArea = 2 * 0.1f * static_cast<float>(mesh.ComputeArea()) / static_cast<float>(mesh.faces.size());
	const float hdiagonal = tree.GetAABBox().sizes().norm() / 2;

	const unsigned itersRay = 50;
	for (unsigned iter = 0; iter < itersRay; ++iter) {
		const unsigned idxFace = RandomRange(0, static_cast<unsigned>(mesh.faces.size()));
		const halfmesh::Mesh::Face& face = mesh.faces[idxFace];
		if (mesh.ComputeFaceDoubleArea(face) < minArea)
			continue;
		const Eigen::Vector3f bary = RandomBarycenter();
		const Eigen::Vector3f X =
		    mesh.vertices[face.x()] * bary.x() + mesh.vertices[face.y()] * bary.y() + mesh.vertices[face.z()] * bary.z();
		const Eigen::Vector3f normal = mesh.ComputeFaceNormal(face).normalized();
		const Eigen::Vector3f origin = X + normal * hdiagonal;
		const Eigen::ParametrizedLine<float, 3> ray(origin, -normal);
		const halfmesh::TriangleKdTree::NearestNeighbor nn = tree.IntersectedPoint(ray);
		halfmesh::Mesh::Type bestDist = std::numeric_limits<halfmesh::Mesh::Type>::max();
		FOREACHIDX (halfmesh::Mesh::FIndex, candidate, mesh.faces) {
			const halfmesh::Mesh::Face& candidateFace = mesh.faces[candidate];
			halfmesh::Mesh::Type dist;
			if (math::RayTriangleIntersect(ray, mesh.vertices[candidateFace[0]],
			                               mesh.vertices[candidateFace[1]], mesh.vertices[candidateFace[2]],
			                               0.f, bestDist, &dist))
				bestDist = dist;
		}
		EXPECT_TRUE(nn.IsValid()) << "iter=" << iter;
		EXPECT_GT(nn.dist, 0.f) << "iter=" << iter;
		EXPECT_NEAR(nn.dist, bestDist, 1e-5f) << "iter=" << iter;
	}
}

// ===========================================================================
// Accuracy regressions
// ===========================================================================
namespace {

// Write text/bytes to a temp file and return its absolute path.
std::string WriteTempText(const std::string& name, const std::string& contents)
{
	const std::string path = (std::filesystem::temp_directory_path() / name).string();
	std::ofstream f(path, std::ios::binary);
	f << contents;
	f.close();
	return path;
}

// Append the raw bytes of a POD vector to a glTF buffer; return the byte offset.
template <typename T>
size_t AppendToBuffer(tinygltf::Buffer& buffer, const std::vector<T>& v)
{
	const size_t off = buffer.data.size();
	const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
	buffer.data.insert(buffer.data.end(), p, p + v.size() * sizeof(T));
	return off;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// LoadPLY must narrow FLOAT64 positions to float instead of memcpy'ing 24-byte
// doubles into 12-byte float slots (garbage geometry + heap overflow in
// NDEBUG/Release builds where the ASSERT type guard compiles out).
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, PLYDoublePrecisionPositions)
{
	const std::string ply =
	    "ply\n"
	    "format ascii 1.0\n"
	    "element vertex 3\n"
	    "property double x\n"
	    "property double y\n"
	    "property double z\n"
	    "element face 1\n"
	    "property list uchar uint vertex_indices\n"
	    "end_header\n"
	    "1.5 2.5 3.5\n"
	    "4.5 5.5 6.5\n"
	    "7.5 8.5 9.5\n"
	    "3 0 1 2\n";
	const std::string path = WriteTempText("halfmesh_double.ply", ply);
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.LoadPLY(path));
	ASSERT_EQ(mesh.vertices.size(), 3u);
	ASSERT_EQ(mesh.faces.size(), 1u);
	EXPECT_NEAR(mesh.vertices[0].x(), 1.5f, 1e-6f);
	EXPECT_NEAR(mesh.vertices[0].y(), 2.5f, 1e-6f);
	EXPECT_NEAR(mesh.vertices[0].z(), 3.5f, 1e-6f);
	EXPECT_NEAR(mesh.vertices[2].x(), 7.5f, 1e-6f);
	EXPECT_NEAR(mesh.vertices[2].y(), 8.5f, 1e-6f);
	EXPECT_NEAR(mesh.vertices[2].z(), 9.5f, 1e-6f);
}

// ---------------------------------------------------------------------------
// PLY colors must respect the file's declared property order (tinyply fills
// buffers in FILE order, not request order).  A standard red,green,blue file
// with red=255 must load as BGR Pixel(0,0,255), not the swapped Pixel(255,0,0).
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, PLYColorChannelOrder)
{
	const std::string ply =
	    "ply\n"
	    "format ascii 1.0\n"
	    "element vertex 3\n"
	    "property float x\n"
	    "property float y\n"
	    "property float z\n"
	    "property uchar red\n"
	    "property uchar green\n"
	    "property uchar blue\n"
	    "element face 1\n"
	    "property list uchar uint vertex_indices\n"
	    "end_header\n"
	    "0 0 0 255 0 0\n"
	    "1 0 0 0 255 0\n"
	    "0 1 0 0 0 255\n"
	    "3 0 1 2\n";
	const std::string path = WriteTempText("halfmesh_color.ply", ply);
	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.LoadPLY(path));
	ASSERT_EQ(mesh.vertexColors.size(), 3u);
	// Pixel is BGR: .x()=blue, .y()=green, .z()=red.
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[0].x()), 0); // vertex 0 red=255
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[0].y()), 0);
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[0].z()), 255);
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[1].x()), 0); // vertex 1 green=255
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[1].y()), 255);
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[1].z()), 0);
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[2].x()), 255); // vertex 2 blue=255
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[2].y()), 0);
	EXPECT_EQ(static_cast<int>(mesh.vertexColors[2].z()), 0);
}

// ---------------------------------------------------------------------------
// A missing/unreadable sidecar texture must NOT collapse every UV to
// (-0.5,-0.5) via a 0x0 texture size; the converters fall back to a 1x1 size.
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, PLYMissingTextureUVFallback)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2)};
	mesh.faceTexcoords = {{0.f, 0.f}, {3.f, 0.f}, {0.f, 3.f}};
	cv::Mat tex(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
	mesh.texturesDiffuse.emplace_back(tex);
	ASSERT_TRUE(mesh.HasTexture());

	const std::string ply = (std::filesystem::temp_directory_path() / "halfmesh_missing_tex.ply").string();
	ASSERT_TRUE(mesh.SavePLY(ply, /*binary=*/true));
	const std::string sidecar =
	    (std::filesystem::temp_directory_path() / "halfmesh_missing_tex_diffuse00.jpg").string();
	ASSERT_TRUE(std::filesystem::exists(sidecar));
	std::filesystem::remove(sidecar);

	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.LoadPLY(ply));
	ASSERT_EQ(reloaded.faceTexcoords.size(), 3u);
	const halfmesh::Mesh::TexCoord collapsed(-0.5f, -0.5f);
	int numCollapsed = 0;
	for (const auto& uv : reloaded.faceTexcoords)
		if ((uv - collapsed).norm() < 1e-4f)
			++numCollapsed;
	EXPECT_LT(numCollapsed, 3)
	    << "all UVs collapsed to (-0.5,-0.5): missing texture => divide-by-zero size";
	EXPECT_GT((reloaded.faceTexcoords[0] - reloaded.faceTexcoords[1]).norm(), 1e-4f);
}

// ---------------------------------------------------------------------------
// A sidecar texture whose file EXISTS but whose bytes fail to DECODE (garbage
// content) must degrade exactly like the missing-file case: the load succeeds,
// the blob stays an empty Mat (warned), and UVs survive via the 1x1 fallback
// size. This pins the decode-side half of the missing/unreadable contract
// (cv::imread returning empty -- or throwing -- inside the parallel decode
// loop), complementing PLYMissingTextureUVFallback's open-failure half.
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, PLYCorruptTextureDecodeDegradesGracefully)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2)};
	mesh.faceTexcoords = {{0.f, 0.f}, {3.f, 0.f}, {0.f, 3.f}};
	cv::Mat tex(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
	mesh.texturesDiffuse.emplace_back(tex);
	ASSERT_TRUE(mesh.HasTexture());

	const std::string ply = (std::filesystem::temp_directory_path() / "halfmesh_corrupt_tex.ply").string();
	ASSERT_TRUE(mesh.SavePLY(ply, /*binary=*/true));
	const std::string sidecar =
	    (std::filesystem::temp_directory_path() / "halfmesh_corrupt_tex_diffuse00.jpg").string();
	ASSERT_TRUE(std::filesystem::exists(sidecar));
	// overwrite the valid JPEG with bytes no image decoder accepts
	WriteTempText("halfmesh_corrupt_tex_diffuse00.jpg", "definitely not a JPEG payload");

	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.LoadPLY(ply));
	ASSERT_EQ(reloaded.texturesDiffuse.size(), 1u);
	EXPECT_TRUE(reloaded.texturesDiffuse[0].empty())
	    << "undecodable sidecar must leave the blob empty, not partially filled";
	ASSERT_EQ(reloaded.faceTexcoords.size(), 3u);
	const halfmesh::Mesh::TexCoord collapsed(-0.5f, -0.5f);
	int numCollapsed = 0;
	for (const auto& uv : reloaded.faceTexcoords)
		if ((uv - collapsed).norm() < 1e-4f)
			++numCollapsed;
	EXPECT_LT(numCollapsed, 3)
	    << "all UVs collapsed to (-0.5,-0.5): decode failure fell into the 0x0-size path";
	EXPECT_GT((reloaded.faceTexcoords[0] - reloaded.faceTexcoords[1]).norm(), 1e-4f);
}

// ---------------------------------------------------------------------------
// An untextured mesh carrying normalized [0,1] atlas UVs (exactly what
// GenerateAtlas produces: faceTexcoords in [0,1], no texture image) must
// stay inside the unit square when written to PLY.  With no texture the
// pixel-space transform used the 1x1 fallback size, writing (u+0.5, 0.5-v)
// — the atlas shifted half a tile and flipped for any external reader — even
// though halfmesh round-tripped it losslessly.  Save must instead pass the
// UVs through (image-space Y-flip only), and Load must invert only that flip.
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, PLYUntexturedNormalizedUVsStayInUnitSquare)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2)};
	// Normalized [0,1] UVs, deliberately near the corners so the buggy
	// +0.5 / 0.5-v transform pushes them outside [0,1].
	mesh.faceTexcoords = {{0.25f, 0.10f}, {0.80f, 0.30f}, {0.40f, 0.95f}};
	ASSERT_TRUE(mesh.HasTextureCoordinates());
	ASSERT_TRUE(mesh.texturesDiffuse.empty());

	// The on-disk (normalized-for-file) UVs must lie in the unit square: an
	// external PLY reader consumes them verbatim.
	const std::vector<halfmesh::Mesh::TexCoord> onDisk = mesh.FTexcoordsNormalizeFlipY();
	for (const halfmesh::Mesh::TexCoord& uv : onDisk) {
		EXPECT_GE(uv.x(), 0.f) << "on-disk U left the unit square";
		EXPECT_LE(uv.x(), 1.f) << "on-disk U left the unit square";
		EXPECT_GE(uv.y(), 0.f) << "on-disk V left the unit square";
		EXPECT_LE(uv.y(), 1.f) << "on-disk V left the unit square";
	}

	// Full save/load round-trip must reproduce the original UVs exactly (the
	// load path stays the exact inverse of save).
	const std::string ply =
	    (std::filesystem::temp_directory_path() / "halfmesh_untex_atlas.ply").string();
	ASSERT_TRUE(mesh.SavePLY(ply, /*binary=*/true));
	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.LoadPLY(ply));
	ASSERT_EQ(reloaded.faceTexcoords.size(), mesh.faceTexcoords.size());
	for (size_t i = 0; i < mesh.faceTexcoords.size(); ++i) {
		EXPECT_LT((reloaded.faceTexcoords[i] - mesh.faceTexcoords[i]).norm(), 1e-4f)
		    << "untextured atlas UV " << i << " did not round-trip";
		EXPECT_GE(reloaded.faceTexcoords[i].x(), 0.f);
		EXPECT_LE(reloaded.faceTexcoords[i].x(), 1.f);
		EXPECT_GE(reloaded.faceTexcoords[i].y(), 0.f);
		EXPECT_LE(reloaded.faceTexcoords[i].y(), 1.f);
	}
}

// ---------------------------------------------------------------------------
// LoadGLTF must keep texture coordinates aligned when an untextured primitive
// precedes a UV-bearing one (backfill earlier faces with (0,0)) rather than
// discarding the whole faceTexcoords array.
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, GLTFUntexturedPrimitiveBeforeUVKeepsTexcoords)
{
	tinygltf::Model model;
	model.asset.version = "2.0";
	tinygltf::Buffer buffer;

	const std::vector<float> aPos{0, 0, 0, 1, 0, 0, 0, 1, 0};
	const std::vector<uint32_t> aIdx{0, 1, 2};
	const std::vector<float> bPos{0, 0, 1, 1, 0, 1, 0, 1, 1};
	const std::vector<float> bUv{0, 0, 1, 0, 0, 1};
	const std::vector<uint32_t> bIdx{0, 1, 2};

	auto addView = [&](size_t off, size_t len) {
		tinygltf::BufferView bv;
		bv.buffer = 0;
		bv.byteOffset = off;
		bv.byteLength = len;
		model.bufferViews.push_back(bv);
		return static_cast<int>(model.bufferViews.size() - 1);
	};
	const int bvAPos = addView(AppendToBuffer(buffer, aPos), aPos.size() * 4);
	const int bvAIdx = addView(AppendToBuffer(buffer, aIdx), aIdx.size() * 4);
	const int bvBPos = addView(AppendToBuffer(buffer, bPos), bPos.size() * 4);
	const int bvBUv = addView(AppendToBuffer(buffer, bUv), bUv.size() * 4);
	const int bvBIdx = addView(AppendToBuffer(buffer, bIdx), bIdx.size() * 4);
	model.buffers.push_back(buffer);

	auto addAcc = [&](int bv, int comp, int type, size_t count) {
		tinygltf::Accessor a;
		a.bufferView = bv;
		a.componentType = comp;
		a.type = type;
		a.count = count;
		model.accessors.push_back(a);
		return static_cast<int>(model.accessors.size() - 1);
	};
	const int accAPos = addAcc(bvAPos, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 3);
	const int accAIdx = addAcc(bvAIdx, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, 3);
	const int accBPos = addAcc(bvBPos, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 3);
	const int accBUv = addAcc(bvBUv, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, 3);
	const int accBIdx = addAcc(bvBIdx, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, 3);

	tinygltf::Mesh gmesh;
	tinygltf::Primitive primA;
	primA.mode = TINYGLTF_MODE_TRIANGLES;
	primA.attributes["POSITION"] = accAPos;
	primA.indices = accAIdx;
	gmesh.primitives.push_back(primA); // untextured first
	tinygltf::Primitive primB;
	primB.mode = TINYGLTF_MODE_TRIANGLES;
	primB.attributes["POSITION"] = accBPos;
	primB.attributes["TEXCOORD_0"] = accBUv;
	primB.indices = accBIdx;
	gmesh.primitives.push_back(primB); // UV-bearing second
	model.meshes.push_back(gmesh);

	tinygltf::Node node;
	node.mesh = 0;
	model.nodes.push_back(node);
	tinygltf::Scene scene;
	scene.nodes.push_back(0);
	model.scenes.push_back(scene);
	model.defaultScene = 0;

	const std::string tmp = (std::filesystem::temp_directory_path() / "halfmesh_untex_then_uv.glb").string();
	tinygltf::TinyGLTF writer;
	ASSERT_TRUE(writer.WriteGltfSceneToFile(&model, tmp, true, true, false, true));

	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.LoadGLTF(tmp));
	EXPECT_EQ(mesh.faces.size(), 2u);
	EXPECT_TRUE(mesh.HasTextureCoordinates())
	    << "UVs from the second primitive were dropped due to untextured-first alignment";
	ASSERT_EQ(mesh.faceTexcoords.size(), mesh.faces.size() * 3);
	// The second primitive contributed non-trivial UVs (>0 somewhere).
	float maxUv = 0.f;
	for (const auto& uv : mesh.faceTexcoords)
		maxUv = std::max(maxUv, std::max(std::abs(uv.x()), std::abs(uv.y())));
	EXPECT_GT(maxUv, 0.5f) << "second primitive's UVs did not survive";
}

// ---------------------------------------------------------------------------
// AccessorData must accept a spec-valid interleaved accessor whose buffer ends
// exactly at the last element (no trailing full stride).  The old check
// required stride*count bytes and dropped such primitives entirely.
// ---------------------------------------------------------------------------
TEST(MeshIoAccuracy, GLTFTailTightStridedAccessorImports)
{
	tinygltf::Model model;
	model.asset.version = "2.0";
	tinygltf::Buffer buffer;
	// 3 vertices, stride 20 (12-byte VEC3 pos + 8 padding), tail-tight: the last
	// vertex has NO trailing padding, so byteLength = 20*2 + 12 = 52.
	const float positions[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
	buffer.data.resize(52, 0);
	for (int i = 0; i < 3; ++i)
		std::memcpy(buffer.data.data() + i * 20, positions[i], 12);
	model.buffers.push_back(buffer);

	tinygltf::BufferView bv;
	bv.buffer = 0;
	bv.byteOffset = 0;
	bv.byteLength = 52;
	bv.byteStride = 20;
	model.bufferViews.push_back(bv);

	tinygltf::Accessor acc;
	acc.bufferView = 0;
	acc.byteOffset = 0;
	acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
	acc.type = TINYGLTF_TYPE_VEC3;
	acc.count = 3;
	acc.minValues = {0, 0, 0};
	acc.maxValues = {1, 1, 0};
	model.accessors.push_back(acc);

	tinygltf::Mesh gmesh;
	tinygltf::Primitive prim;
	prim.mode = TINYGLTF_MODE_TRIANGLES;
	prim.attributes["POSITION"] = 0;
	prim.indices = -1; // implicit 0,1,2
	gmesh.primitives.push_back(prim);
	model.meshes.push_back(gmesh);

	tinygltf::Node node;
	node.mesh = 0;
	model.nodes.push_back(node);
	tinygltf::Scene scene;
	scene.nodes.push_back(0);
	model.scenes.push_back(scene);
	model.defaultScene = 0;

	const std::string tmp = (std::filesystem::temp_directory_path() / "halfmesh_tailtight.glb").string();
	tinygltf::TinyGLTF writer;
	ASSERT_TRUE(writer.WriteGltfSceneToFile(&model, tmp, true, true, false, true));

	halfmesh::Mesh mesh;
	ASSERT_TRUE(mesh.LoadGLTF(tmp))
	    << "spec-valid tail-tight interleaved accessor was rejected";
	EXPECT_EQ(mesh.vertices.size(), 3u);
	EXPECT_EQ(mesh.faces.size(), 1u);
}

// The loaders parse untrusted files; failures must come back as `false`
// through the bool API, never as exceptions or out-of-bounds reads

// tinyply throws on a face list that differs from the triangle size hint —
// a quad PLY (Blender's default export) must fail gracefully, not crash.
TEST(MeshIoTest, QuadPlyFailsGracefully)
{
	const std::string path =
	    (std::filesystem::temp_directory_path() / "halfmesh_quad.ply").string();
	{
		std::ofstream f(path);
		f << "ply\nformat ascii 1.0\n"
		     "element vertex 4\n"
		     "property float x\nproperty float y\nproperty float z\n"
		     "element face 1\n"
		     "property list uchar int vertex_indices\n"
		     "end_header\n"
		     "0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
		     "4 0 1 2 3\n";
	}
	halfmesh::Mesh mesh;
	EXPECT_FALSE(mesh.Load(path));
	std::filesystem::remove(path);
}

// A truncated binary PLY (partial download / interrupted write) must also
// come back as false, not as an escaping std::runtime_error.
TEST(MeshIoTest, TruncatedPlyFailsGracefully)
{
	const std::string path =
	    (std::filesystem::temp_directory_path() / "halfmesh_trunc.ply").string();
	{
		halfmesh::Mesh mesh;
		mesh.vertices = {halfmesh::Mesh::Vertex(0, 0, 0), halfmesh::Mesh::Vertex(1, 0, 0),
		                 halfmesh::Mesh::Vertex(0, 1, 0), halfmesh::Mesh::Vertex(0, 0, 1)};
		mesh.faces = {halfmesh::Mesh::Face(0, 1, 2), halfmesh::Mesh::Face(1, 0, 3),
		              halfmesh::Mesh::Face(2, 1, 3), halfmesh::Mesh::Face(0, 2, 3)};
		ASSERT_TRUE(mesh.SavePLY(path, /*binary=*/true));
	}
	// Find the header/body boundary to truncate inside binary data (not the header).
	std::ifstream f(path, std::ios::binary);
	std::string headerLine;
	size_t headerEnd = 0;
	while (std::getline(f, headerLine)) {
		headerEnd += headerLine.size() + 1; // +1 for the newline that getline consumes
		if (headerLine == "end_header") {
			break;
		}
	}
	f.close();
	const auto full = std::filesystem::file_size(path);
	// Truncate to halfway through the binary body (after headerEnd).
	const size_t truncateAt = headerEnd + (full - headerEnd) / 2;
	std::filesystem::resize_file(path, truncateAt);
	halfmesh::Mesh mesh;
	EXPECT_FALSE(mesh.Load(path));
	std::filesystem::remove(path);
}

// glTF index values are file-supplied: an out-of-range entry must not be
// subscripted into the UV array unchecked (OOB read) or plant a wild vertex
// index in faces — the primitive must be skipped instead.
TEST(MeshIoTest, GltfOutOfRangeIndicesRejected)
{
	namespace fs = std::filesystem;
	const fs::path dir = fs::temp_directory_path();
	const std::string binName = "halfmesh_oob.bin";
	const std::string gltfPath = (dir / "halfmesh_oob.gltf").string();
	{
		// buffer: 3 vec3f positions (36 B) + 3 vec2f UVs (24 B) + 3 uint32 indices (12 B)
		std::ofstream b(dir / binName, std::ios::binary);
		const float pos[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
		const float uv[6] = {0, 0, 1, 0, 0, 1};
		const uint32_t idx[3] = {0, 1, 7}; // 7 >= vertex count 3: out of range
		b.write(reinterpret_cast<const char*>(pos), sizeof(pos));
		b.write(reinterpret_cast<const char*>(uv), sizeof(uv));
		b.write(reinterpret_cast<const char*>(idx), sizeof(idx));
	}
	{
		std::ofstream g(gltfPath);
		g << "{\"asset\":{\"version\":\"2.0\"},"
		     "\"scenes\":[{\"nodes\":[0]}],\"scene\":0,"
		     "\"nodes\":[{\"mesh\":0}],"
		     "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":1},\"indices\":2}]}],"
		     "\"accessors\":["
		     "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\",\"min\":[0,0,0],\"max\":[1,1,0]},"
		     "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
		     "{\"bufferView\":2,\"componentType\":5125,\"count\":3,\"type\":\"SCALAR\"}],"
		     "\"bufferViews\":["
		     "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
		     "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":24},"
		     "{\"buffer\":0,\"byteOffset\":60,\"byteLength\":12}],"
		     "\"buffers\":[{\"uri\":\""
		  << binName << "\",\"byteLength\":72}]}";
	}
	halfmesh::Mesh mesh;
	// the only primitive is skipped -> no triangle geometry -> Load fails
	EXPECT_FALSE(mesh.Load(gltfPath));
	fs::remove(gltfPath);
	fs::remove(dir / binName);
}

// SavePLY must survive an empty texturesDiffuse entry — exactly what LoadPLY
// leaves in place for a missing sidecar texture: cv::imwrite throws on an
// empty Mat, and under ParallelForPool's rethrow-after-wait() contract that
// escaped SavePLY's bool API until the per-item catch (2026-07-17 final batch
// review). One bad texture must not abort its siblings or the geometry write.
TEST(MeshIoTest, SavePLYEmptyTextureDegradesGracefully)
{
	halfmesh::Mesh mesh;
	mesh.vertices = {
	    halfmesh::Mesh::Vertex(0.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(1.f, 0.f, 0.f),
	    halfmesh::Mesh::Vertex(0.f, 1.f, 0.f),
	};
	mesh.faces = {halfmesh::Mesh::Face(0, 1, 2)};
	mesh.faceTexcoords = {{0.f, 0.f}, {3.f, 0.f}, {0.f, 3.f}};
	cv::Mat validTex(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
	mesh.texturesDiffuse.emplace_back(validTex);
	mesh.texturesDiffuse.emplace_back(); // empty Mat: the missing-texture case
	ASSERT_TRUE(mesh.HasTexture());

	const std::filesystem::path dir = std::filesystem::temp_directory_path();
	const std::string ply = (dir / "halfmesh_empty_tex.ply").string();
	const std::string sidecar0 = (dir / "halfmesh_empty_tex_diffuse00.jpg").string();
	const std::string sidecar1 = (dir / "halfmesh_empty_tex_diffuse01.jpg").string();
	std::filesystem::remove(sidecar0); // stale files from prior runs
	std::filesystem::remove(sidecar1);

	ASSERT_TRUE(mesh.SavePLY(ply, /*binary=*/true))
	    << "an empty texture must degrade (skip + warn), not fail the save";
	EXPECT_TRUE(std::filesystem::exists(sidecar0))
	    << "the valid sibling texture must still be written";
	EXPECT_FALSE(std::filesystem::exists(sidecar1))
	    << "the empty texture must be skipped, not written";

	halfmesh::Mesh reloaded;
	ASSERT_TRUE(reloaded.LoadPLY(ply)) << "geometry must round-trip";
	EXPECT_EQ(reloaded.faces.size(), 1u);
	EXPECT_EQ(reloaded.faceTexcoords.size(), 3u);
}
