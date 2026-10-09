/*
* MeshIOOBJ.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Implements: Mesh::LoadOBJ / Mesh::SaveOBJ -- Wavefront OBJ with its MTL library.
//
// Reading follows the architecture of rapidobj, the fastest OBJ parser measured
// (Aras Pranckevicius, "Comparing .obj parse libraries", 2022): the file streams
// in large blocks, the next one read while the current one parses, and each block
// is parsed in parallel over newline-aligned chunks. Here every block takes two
// passes so that each element lands straight in its final slot: a counting pass
// sizes every chunk, a prefix sum places it, and the parsing pass resolves the
// relative indices against the exact global counts. Floats are parsed by
// fast_float (correctly rounded; the algorithm behind std::from_chars in
// libstdc++ and MSVC) and written by std::to_chars in their shortest round-trip
// form, so a Save -> Load cycle is bit-exact.

#include <halfmesh/Mesh.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h>
#include <halfmesh/Version.h>

#include <opencv2/imgcodecs.hpp>
#include <fast_float/fast_float.h>
#include <BS_thread_pool.hpp>

#include "ParallelFor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace halfmesh {

namespace {

using detail::ParallelForPool;

constexpr size_t OBJ_BLOCK_BYTES = size_t(32) << 20; // streamed read unit
constexpr size_t OBJ_CHUNK_BYTES = size_t(1) << 20; // parallel parse unit
constexpr uint32_t NO_INDEX = math::NO_ID;

// --- tokenizing ------------------------------------------------------------

inline bool IsBlank(char c)
{
	return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}
inline const char* SkipBlanks(const char* p, const char* end)
{
	while (p < end && IsBlank(*p))
		++p;
	return p;
}
inline const char* SkipToken(const char* p, const char* end)
{
	while (p < end && !IsBlank(*p))
		++p;
	return p;
}
// the line up to a comment, without trailing blanks
inline const char* StripComment(const char* first, const char* last)
{
	const char* hash = static_cast<const char*>(std::memchr(first, '#', static_cast<size_t>(last - first)));
	if (hash)
		last = hash;
	while (last > first && IsBlank(last[-1]))
		--last;
	return last;
}
inline bool IsKeyword(const char* p, const char* end, std::string_view keyword)
{
	const size_t n = keyword.size();
	return static_cast<size_t>(end - p) >= n && std::memcmp(p, keyword.data(), n) == 0 && (p + n == end || IsBlank(p[n]));
}
inline bool ParseFloat(const char*& p, const char* end, float& value)
{
	p = SkipBlanks(p, end);
	const fast_float::parse_options options{fast_float::chars_format::general | fast_float::chars_format::allow_leading_plus};
	const auto result = fast_float::from_chars_advanced(p, end, value, options);
	if (result.ec != std::errc() || (result.ptr < end && !IsBlank(*result.ptr)))
		return false;
	p = result.ptr;
	return true;
}

// Iterates the logical lines of [p, end): a physical line ending in a backslash
// continues on the next one. Those are rare, so they are joined in a scratch
// buffer and everything else is handed out in place.
struct LineReader
{
	const char* p;
	const char* end;
	size_t numPhysical{0}; // physical lines consumed so far
	std::string joined;

	LineReader(const char* first, const char* last) :
	    p(first), end(last) {}

	bool Next(const char*& first, const char*& last)
	{
		if (p >= end)
			return false;
		ReadPhysical(first, last);
		if (last == first || last[-1] != '\\')
			return true;
		joined.assign(first, last - 1);
		while (p < end) {
			const char *nextFirst, *nextLast;
			ReadPhysical(nextFirst, nextLast);
			joined.push_back(' ');
			if (nextLast > nextFirst && nextLast[-1] == '\\') {
				joined.append(nextFirst, nextLast - 1);
				continue;
			}
			joined.append(nextFirst, nextLast);
			break;
		}
		first = joined.data();
		last = first + joined.size();
		return true;
	}

	private:
	void ReadPhysical(const char*& first, const char*& last)
	{
		const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
		first = p;
		last = nl ? nl : end;
		p = nl ? nl + 1 : end;
		++numPhysical;
		if (last > first && last[-1] == '\r')
			--last;
	}
};

// One past the newline that ends the logical line holding `pos`, or `end`
// (`begin` bounds the look-back for the continuation backslash).
const char* LogicalLineEnd(const char* begin, const char* pos, const char* end)
{
	while (pos < end) {
		const char* nl = static_cast<const char*>(std::memchr(pos, '\n', static_cast<size_t>(end - pos)));
		if (!nl)
			return end;
		const char* q = nl;
		if (q > begin && q[-1] == '\r')
			--q;
		if (q == begin || q[-1] != '\\')
			return nl + 1;
		pos = nl + 1;
	}
	return end;
}

// --- one parsed chunk ------------------------------------------------------

// a polygon corner, indices resolved to 0-based (NO_INDEX when absent)
struct Corner
{
	uint32_t v, vt, vn;
};
// a polygon with more than three corners, triangulated once every position is known
struct LargePolygon
{
	size_t firstTriangle; // its slots in the triangle arrays
	size_t firstCorner; // into the large-polygon corner array
	uint32_t numCorners;
	uint32_t material;
};

struct Chunk
{
	const char* first;
	const char* last;
	// counting pass
	size_t numV{0}, numVT{0}, numVN{0}, numColored{0};
	size_t numTriangles{0}, numLarge{0}, numLargeCorners{0}, numLines{0};
	bool hasVT{false}, hasVN{false};
	std::vector<std::string> materials; // usemtl names, in order
	std::vector<std::string> libraries; // mtllib arguments
	// placement
	size_t vBase{0}, vtBase{0}, vnBase{0}, triangleBase{0}, largeBase{0}, largeCornerBase{0}, lineBase{0};
	uint32_t material{0}; // the material in effect where the chunk starts
	// parsing pass
	size_t numSkipped{0}; // polygons with fewer than three corners
	size_t errorLine{0}; // physical line, 1-based within the chunk; 0 = no error
	std::string error;
};

// the rest of a usemtl / mtllib statement
std::string StatementArgument(const char* p, const char* end)
{
	p = SkipBlanks(p, end);
	return std::string(p, end);
}

void CountChunk(Chunk& chunk)
{
	LineReader reader(chunk.first, chunk.last);
	const char *first, *last;
	while (reader.Next(first, last)) {
		last = StripComment(first, last);
		const char* p = SkipBlanks(first, last);
		if (p == last)
			continue;
		if (p[0] == 'v') {
			if (IsKeyword(p, last, "v")) {
				++chunk.numV;
				unsigned numValues = 0;
				for (p = SkipBlanks(p + 1, last); p < last; p = SkipBlanks(SkipToken(p, last), last))
					++numValues;
				if (numValues >= 6)
					++chunk.numColored;
			} else if (IsKeyword(p, last, "vt")) {
				++chunk.numVT;
			} else if (IsKeyword(p, last, "vn")) {
				++chunk.numVN;
			}
		} else if (IsKeyword(p, last, "f")) {
			uint32_t numCorners = 0;
			for (p = SkipBlanks(p + 1, last); p < last; ++numCorners) {
				const char* tokenEnd = SkipToken(p, last);
				const char* slash = static_cast<const char*>(std::memchr(p, '/', static_cast<size_t>(tokenEnd - p)));
				if (slash) {
					if (slash + 1 < tokenEnd && slash[1] != '/')
						chunk.hasVT = true;
					if (std::memchr(slash + 1, '/', static_cast<size_t>(tokenEnd - slash - 1)))
						chunk.hasVN = true;
				}
				p = SkipBlanks(tokenEnd, last);
			}
			if (numCorners >= 3) {
				chunk.numTriangles += numCorners - 2;
				if (numCorners > 3) {
					++chunk.numLarge;
					chunk.numLargeCorners += numCorners;
				}
			}
		} else if (IsKeyword(p, last, "usemtl")) {
			chunk.materials.emplace_back(StatementArgument(p + 6, last));
		} else if (IsKeyword(p, last, "mtllib")) {
			chunk.libraries.emplace_back(StatementArgument(p + 6, last));
		}
	}
	chunk.numLines = reader.numPhysical;
}

// where the parsing pass writes; the arrays are sized for the block beforehand
struct Targets
{
	Mesh::Vertex* positions;
	Eigen::Vector3f* colors; // null while no vertex has a color
	Mesh::TexCoord* texcoords;
	Mesh::Normal* normals;
	Mesh::Face* triangles;
	uint32_t* cornerVT; // three per triangle; null while no corner has a vt
	uint32_t* cornerVN;
	uint32_t* triangleMaterials; // null while every face uses the default material
	LargePolygon* largePolygons;
	Corner* largeCorners;
	const std::unordered_map<std::string, uint32_t>* materialIds;
};

// resolve a 1-based OBJ index (negative: counted back from `count`, the number
// of such elements defined so far) to 0-based; range against the total is
// checked once the whole file is read
inline bool ResolveIndex(int64_t idx, size_t count, uint32_t& resolved)
{
	if (idx > 0) {
		if (idx > int64_t(NO_INDEX))
			return false;
		resolved = static_cast<uint32_t>(idx - 1);
		return true;
	}
	if (idx < 0 && -idx <= static_cast<int64_t>(count)) {
		resolved = static_cast<uint32_t>(static_cast<int64_t>(count) + idx);
		return true;
	}
	return false;
}
inline bool ParseIndex(const char*& p, const char* end, int64_t& idx)
{
	const auto result = std::from_chars(p, end, idx);
	if (result.ec != std::errc())
		return false;
	p = result.ptr;
	return true;
}

void ParseChunk(Chunk& chunk, const Targets& targets)
{
	LineReader reader(chunk.first, chunk.last);
	size_t numV = 0, numVT = 0, numVN = 0, numTriangles = 0, numLarge = 0, numLargeCorners = 0;
	uint32_t material = chunk.material;
	std::vector<Corner> corners;
	const auto fail = [&](const char* message) {
		chunk.errorLine = reader.numPhysical;
		chunk.error = message;
	};
	const char *first, *last;
	while (reader.Next(first, last)) {
		last = StripComment(first, last);
		const char* p = SkipBlanks(first, last);
		if (p == last)
			continue;
		if (p[0] == 'v') {
			if (IsKeyword(p, last, "v")) {
				float values[7];
				unsigned numValues = 0;
				for (p = SkipBlanks(p + 1, last); p < last && numValues < 7; p = SkipBlanks(p, last))
					if (!ParseFloat(p, last, values[numValues++]))
						return fail("invalid vertex");
				if (numValues < 3)
					return fail("vertex with fewer than 3 coordinates");
				const size_t idx = chunk.vBase + numV++;
				targets.positions[idx] = Mesh::Vertex(values[0], values[1], values[2]);
				if (targets.colors)
					targets.colors[idx] = numValues >= 6 ? Eigen::Vector3f(values[numValues - 3], values[numValues - 2], values[numValues - 1]) : Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
			} else if (IsKeyword(p, last, "vt")) {
				float values[2] = {0, 0};
				p += 2;
				for (unsigned i = 0; i < 2 && SkipBlanks(p, last) < last; ++i)
					if (!ParseFloat(p, last, values[i]))
						return fail("invalid texture coordinate");
				targets.texcoords[chunk.vtBase + numVT++] = Mesh::TexCoord(values[0], values[1]);
			} else if (IsKeyword(p, last, "vn")) {
				float values[3];
				p += 2;
				for (float& value : values)
					if (!ParseFloat(p, last, value))
						return fail("invalid normal");
				targets.normals[chunk.vnBase + numVN++] = Mesh::Normal(values[0], values[1], values[2]);
			}
		} else if (IsKeyword(p, last, "f")) {
			corners.clear();
			for (p = SkipBlanks(p + 1, last); p < last; p = SkipBlanks(p, last)) {
				Corner corner{NO_INDEX, NO_INDEX, NO_INDEX};
				int64_t idx;
				if (!ParseIndex(p, last, idx) || !ResolveIndex(idx, chunk.vBase + numV, corner.v))
					return fail("invalid face vertex index");
				if (p < last && *p == '/') {
					++p;
					if (p < last && *p != '/' && !IsBlank(*p)) {
						if (!ParseIndex(p, last, idx) || !ResolveIndex(idx, chunk.vtBase + numVT, corner.vt))
							return fail("invalid face texture coordinate index");
					}
					if (p < last && *p == '/') {
						++p;
						if (p < last && !IsBlank(*p) && (!ParseIndex(p, last, idx) || !ResolveIndex(idx, chunk.vnBase + numVN, corner.vn)))
							return fail("invalid face normal index");
					}
				}
				if (p < last && !IsBlank(*p))
					return fail("invalid face corner");
				corners.push_back(corner);
			}
			if (corners.size() < 3) {
				++chunk.numSkipped;
				continue;
			}
			if (corners.size() == 3) {
				const size_t t = chunk.triangleBase + numTriangles++;
				targets.triangles[t] = Mesh::Face(corners[0].v, corners[1].v, corners[2].v);
				for (int k = 0; k < 3; ++k) {
					if (targets.cornerVT)
						targets.cornerVT[t * 3 + k] = corners[k].vt;
					if (targets.cornerVN)
						targets.cornerVN[t * 3 + k] = corners[k].vn;
				}
				if (targets.triangleMaterials)
					targets.triangleMaterials[t] = material;
			} else {
				const size_t firstCorner = chunk.largeCornerBase + numLargeCorners;
				targets.largePolygons[chunk.largeBase + numLarge++] = LargePolygon{chunk.triangleBase + numTriangles, firstCorner, static_cast<uint32_t>(corners.size()), material};
				std::copy(corners.begin(), corners.end(), targets.largeCorners + firstCorner);
				numLargeCorners += corners.size();
				numTriangles += corners.size() - 2;
			}
		} else if (IsKeyword(p, last, "usemtl")) {
			material = targets.materialIds->at(StatementArgument(p + 6, last));
		}
	}
	ASSERT(numV == chunk.numV && numVT == chunk.numVT && numVN == chunk.numVN && numTriangles == chunk.numTriangles);
}

// --- polygon triangulation -------------------------------------------------

// Split a polygon of n > 3 corners into n-2 triangles (as corner index triples).
// A quad takes the diagonal whose two halves both face the polygon's (Newell)
// normal, the shorter diagonal when both do; a larger polygon is ear-clipped in
// its best-fit plane, and falls back to a fan from the remaining corners when
// no ear is left (a self-intersecting or degenerate outline).
void TriangulatePolygon(const std::vector<Mesh::Vertex>& positions, const Corner* corners, uint32_t n, std::vector<std::array<uint32_t, 3>>& triangles)
{
	triangles.clear();
	const auto position = [&](uint32_t i) -> Eigen::Vector3d { return positions[corners[i].v].cast<double>(); };
	Eigen::Vector3d normal = Eigen::Vector3d::Zero();
	for (uint32_t i = 0; i < n; ++i)
		normal += position(i).cross(position((i + 1) % n));
	if (n == 4) {
		const auto facesNormal = [&](uint32_t a, uint32_t b, uint32_t c) {
			return (position(b) - position(a)).cross(position(c) - position(a)).dot(normal) > 0;
		};
		const bool valid02 = facesNormal(0, 1, 2) && facesNormal(0, 2, 3);
		const bool valid13 = facesNormal(1, 2, 3) && facesNormal(1, 3, 0);
		const bool split02 = valid02 != valid13 ? valid02 : (position(2) - position(0)).squaredNorm() <= (position(3) - position(1)).squaredNorm();
		if (split02) {
			triangles.push_back({0, 1, 2});
			triangles.push_back({0, 2, 3});
		} else {
			triangles.push_back({1, 2, 3});
			triangles.push_back({1, 3, 0});
		}
		return;
	}
	// project on the best-fit plane, oriented so the outline winds counter-clockwise
	std::vector<Eigen::Vector2d> points(n);
	const double normalNorm = normal.norm();
	std::vector<uint32_t> remaining(n);
	for (uint32_t i = 0; i < n; ++i)
		remaining[i] = i;
	if (normalNorm > 0) {
		const Eigen::Vector3d axisZ = normal / normalNorm;
		const Eigen::Vector3d axisX = axisZ.unitOrthogonal();
		const Eigen::Vector3d axisY = axisZ.cross(axisX);
		for (uint32_t i = 0; i < n; ++i)
			points[i] = Eigen::Vector2d(position(i).dot(axisX), position(i).dot(axisY));
		const auto cross = [&](uint32_t a, uint32_t b, uint32_t c) {
			const Eigen::Vector2d ab = points[b] - points[a], ac = points[c] - points[a];
			return ab.x() * ac.y() - ab.y() * ac.x();
		};
		const auto inside = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t q) {
			return cross(a, b, q) >= 0 && cross(b, c, q) >= 0 && cross(c, a, q) >= 0;
		};
		size_t start = 0;
		while (remaining.size() > 3) {
			const size_t m = remaining.size();
			bool clipped = false;
			for (size_t step = 0; step < m && !clipped; ++step) {
				const size_t i = (start + step) % m;
				const uint32_t a = remaining[(i + m - 1) % m], b = remaining[i], c = remaining[(i + 1) % m];
				if (cross(a, b, c) <= 0)
					continue;
				bool ear = true;
				for (const uint32_t q : remaining)
					if (q != a && q != b && q != c && inside(a, b, c, q)) {
						ear = false;
						break;
					}
				if (!ear)
					continue;
				triangles.push_back({a, b, c});
				remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(i));
				start = i % remaining.size();
				clipped = true;
			}
			if (!clipped)
				break;
		}
	}
	for (size_t i = 1; i + 1 < remaining.size(); ++i)
		triangles.push_back({remaining[0], remaining[i], remaining[i + 1]});
	ASSERT(triangles.size() == n - 2);
}

// --- MTL -------------------------------------------------------------------

struct Material
{
	Eigen::Vector3f Kd{1, 1, 1};
	std::filesystem::path map; // empty: no diffuse map
};

inline bool EqualsNoCase(std::string_view a, std::string_view b)
{
	return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
		       return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
	       });
}

// A path as written in an OBJ/MTL statement, relative to `directory`: quotes
// stripped, and a Windows path written with backslashes still found elsewhere.
std::filesystem::path ResolvePath(const std::filesystem::path& directory, std::string name)
{
	if (name.size() >= 2 && name.front() == '"' && name.back() == '"')
		name = name.substr(1, name.size() - 2);
	std::filesystem::path path(name);
	if (path.is_relative())
		path = directory / path;
	std::error_code ec;
	if (!std::filesystem::exists(path, ec) && name.find('\\') != std::string::npos) {
		std::replace(name.begin(), name.end(), '\\', '/');
		std::filesystem::path slashed(name);
		if (slashed.is_relative())
			slashed = directory / slashed;
		if (std::filesystem::exists(slashed, ec))
			return slashed;
	}
	return path;
}

// the file name of a map_* statement, its options skipped
std::string MapFileName(const char* p, const char* end)
{
	for (p = SkipBlanks(p, end); p < end && *p == '-'; p = SkipBlanks(p, end)) {
		const char* optionEnd = SkipToken(p, end);
		const std::string_view option(p, static_cast<size_t>(optionEnd - p));
		p = SkipBlanks(optionEnd, end);
		if (option == "-o" || option == "-s" || option == "-t") {
			// up to three numbers
			for (int i = 0; i < 3 && p < end; ++i) {
				float value;
				const char* q = p;
				if (!ParseFloat(q, end, value))
					break;
				p = SkipBlanks(q, end);
			}
		} else if (option == "-mm") {
			p = SkipBlanks(SkipToken(SkipBlanks(SkipToken(p, end), end), end), end);
		} else {
			// -blendu -blendv -cc -clamp -imfchan -texres -bm -boost -type: one argument
			p = SkipToken(p, end);
		}
	}
	return std::string(p, end);
}

void LoadMaterialLibrary(const std::filesystem::path& fileName, std::unordered_map<std::string, Material>& materials)
{
	std::ifstream file(fileName, std::ios::binary);
	if (!file) {
		REPORT_WARNING("material library '{}' not found", fileName.string());
		return;
	}
	const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	const std::filesystem::path directory(fileName.parent_path());
	LineReader reader(text.data(), text.data() + text.size());
	Material* material = nullptr;
	const char *first, *last;
	while (reader.Next(first, last)) {
		last = StripComment(first, last);
		const char* p = SkipBlanks(first, last);
		const char* keywordEnd = SkipToken(p, last);
		const std::string_view keyword(p, static_cast<size_t>(keywordEnd - p));
		if (EqualsNoCase(keyword, "newmtl")) {
			// the first definition of a name wins, as for the libraries' order
			const auto [it, inserted] = materials.try_emplace(StatementArgument(keywordEnd, last));
			material = inserted ? &it->second : nullptr;
		} else if (material && EqualsNoCase(keyword, "Kd")) {
			p = keywordEnd;
			Eigen::Vector3f Kd;
			if (ParseFloat(p, last, Kd[0]) && ParseFloat(p, last, Kd[1]) && ParseFloat(p, last, Kd[2]))
				material->Kd = Kd;
		} else if (material && EqualsNoCase(keyword, "map_Kd")) {
			const std::string name = MapFileName(keywordEnd, last);
			if (!name.empty())
				material->map = ResolvePath(directory, name);
		}
	}
}

// --- writing ---------------------------------------------------------------

inline void AppendFloat(std::string& text, float value)
{
	char buffer[32];
	const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
	text.append(buffer, result.ptr);
}
inline void AppendIndex(std::string& text, uint32_t idx)
{
	char buffer[16];
	const auto result = std::to_chars(buffer, buffer + sizeof(buffer), uint64_t(idx) + 1);
	text.append(buffer, result.ptr);
}

// Format `count` lines in parallel blocks and write them in order; the blocks go
// out in waves, so the text held at once stays bounded whatever the mesh size.
template <typename FormatLine>
void WriteLines(std::ofstream& out, BS::light_thread_pool& pool, size_t count, FormatLine&& formatLine)
{
	constexpr size_t LINES_PER_BLOCK = size_t(1) << 15;
	const size_t numBlocks = (count + LINES_PER_BLOCK - 1) / LINES_PER_BLOCK;
	const size_t blocksPerWave = std::max<size_t>(pool.get_thread_count() * 4, 1);
	std::vector<std::string> texts(std::min(numBlocks, blocksPerWave));
	for (size_t wave = 0; wave < numBlocks; wave += blocksPerWave) {
		const size_t numWaveBlocks = std::min(blocksPerWave, numBlocks - wave);
		ParallelForPool(pool, numWaveBlocks, [&](size_t i) {
			std::string& text = texts[i];
			text.clear();
			const size_t begin = (wave + i) * LINES_PER_BLOCK;
			const size_t end = std::min(count, begin + LINES_PER_BLOCK);
			for (size_t line = begin; line < end; ++line)
				formatLine(text, line);
		});
		for (size_t i = 0; i < numWaveBlocks; ++i)
			out.write(texts[i].data(), static_cast<std::streamsize>(texts[i].size()));
	}
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// LoadOBJ
// ---------------------------------------------------------------------------
bool Mesh::LoadOBJ(const std::string& fileName)
{
	vertices.clear();
	faces.clear();
	halfMesh.Clear();
	ReleaseOptional();
	std::ifstream file(fileName, std::ios::binary);
	if (!file) {
		REPORT_WARNING("failed to open {}", fileName);
		return false;
	}
	file.seekg(0, std::ios::end);
	const size_t fileSize = static_cast<size_t>(file.tellg());
	file.seekg(0, std::ios::beg);
	std::error_code canonEc;
	std::filesystem::path canon = std::filesystem::canonical(std::filesystem::path(fileName), canonEc);
	if (canonEc)
		canon = std::filesystem::path(fileName);
	const std::filesystem::path directory(canon.parent_path());

	// the whole file's elements, filled block by block
	std::vector<Vertex> positions;
	std::vector<Eigen::Vector3f> colors;
	std::vector<TexCoord> texcoords;
	std::vector<Normal> normals;
	std::vector<Face> triangles;
	std::vector<uint32_t> cornerVT, cornerVN, triangleMaterials;
	std::vector<LargePolygon> largePolygons;
	std::vector<Corner> largeCorners;
	size_t numColored = 0, numSkipped = 0, numLines = 0;
	bool hasVT = false, hasVN = false, hasMaterials = false;
	std::unordered_map<std::string, uint32_t> materialIds{{std::string(), 0}};
	std::vector<std::string> materialNames{std::string()};
	std::vector<std::string> libraries;
	uint32_t material = 0;

	// grow an array to `size`, by at least half its capacity when it must move
	const auto grow = [](auto& array, size_t size, const auto& value) {
		if (size > array.capacity())
			array.reserve(std::max(size, array.capacity() + array.capacity() / 2));
		array.resize(size, value);
	};

	BS::light_thread_pool pool;
	const auto readBlock = [&file](std::vector<char>& block) {
		block.resize(OBJ_BLOCK_BYTES);
		file.read(block.data(), static_cast<std::streamsize>(block.size()));
		block.resize(static_cast<size_t>(file.gcount()));
	};
	std::vector<char> data, next;
	readBlock(data);
	std::future<void> pending;
	if (file)
		pending = std::async(std::launch::async, readBlock, std::ref(next));
	size_t bytesDone = 0;
	while (!data.empty()) {
		// the block ends with its last complete logical line; the rest carries over
		const bool lastBlock = !pending.valid();
		const char* const begin = data.data();
		const char* end = begin + data.size();
		if (!lastBlock) {
			const char* cut = end;
			while (cut > begin && cut[-1] != '\n')
				--cut;
			// a line continued past the cut belongs to the next block
			while (cut > begin) {
				const char* q = cut - 1;
				if (q > begin && q[-1] == '\r')
					--q;
				if (q == begin || q[-1] != '\\')
					break;
				for (cut = q; cut > begin && cut[-1] != '\n';)
					--cut;
			}
			end = cut;
		}
		// chunks of about OBJ_CHUNK_BYTES, each ending with a logical line
		std::vector<Chunk> chunks;
		for (const char* p = begin; p < end;) {
			const char* q = p + std::min<size_t>(OBJ_CHUNK_BYTES, static_cast<size_t>(end - p));
			if (q < end)
				q = LogicalLineEnd(begin, q, end);
			Chunk& chunk = chunks.emplace_back();
			chunk.first = p;
			chunk.last = q;
			p = q;
		}
		ParallelForPool(pool, chunks.size(), [&](size_t i) { CountChunk(chunks[i]); });

		// place every chunk and collect what the parsing pass needs
		size_t numV = positions.size(), numVT = texcoords.size(), numVN = normals.size();
		size_t numTriangles = triangles.size(), numLarge = largePolygons.size(), numLargeCorners = largeCorners.size();
		bool blockColored = false;
		for (Chunk& chunk : chunks) {
			chunk.vBase = numV;
			chunk.vtBase = numVT;
			chunk.vnBase = numVN;
			chunk.triangleBase = numTriangles;
			chunk.largeBase = numLarge;
			chunk.largeCornerBase = numLargeCorners;
			chunk.lineBase = numLines;
			chunk.material = material;
			numV += chunk.numV;
			numVT += chunk.numVT;
			numVN += chunk.numVN;
			numTriangles += chunk.numTriangles;
			numLarge += chunk.numLarge;
			numLargeCorners += chunk.numLargeCorners;
			numLines += chunk.numLines;
			numColored += chunk.numColored;
			blockColored = blockColored || chunk.numColored != 0;
			hasVT = hasVT || chunk.hasVT;
			hasVN = hasVN || chunk.hasVN;
			for (std::string& name : chunk.materials) {
				const auto [it, inserted] = materialIds.try_emplace(name, static_cast<uint32_t>(materialNames.size()));
				if (inserted)
					materialNames.push_back(std::move(name));
				material = it->second;
				hasMaterials = hasMaterials || material != 0;
			}
			for (std::string& library : chunk.libraries)
				libraries.push_back(std::move(library));
		}
		if (bytesDone == 0 && !lastBlock && end > begin) {
			// size the arrays for the whole file from its first block
			const double scale = 1.05 * static_cast<double>(fileSize) / static_cast<double>(end - begin);
			const auto estimate = [scale](size_t count) { return static_cast<size_t>(static_cast<double>(count) * scale); };
			positions.reserve(estimate(numV));
			texcoords.reserve(estimate(numVT));
			normals.reserve(estimate(numVN));
			triangles.reserve(estimate(numTriangles));
		}
		grow(positions, numV, Vertex::Zero().eval());
		if (blockColored || !colors.empty())
			grow(colors, numV, Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()).eval());
		grow(texcoords, numVT, TexCoord::Zero().eval());
		grow(normals, numVN, Normal::Zero().eval());
		grow(triangles, numTriangles, Face::Zero().eval());
		if (hasVT)
			grow(cornerVT, numTriangles * 3, NO_INDEX);
		if (hasVN)
			grow(cornerVN, numTriangles * 3, NO_INDEX);
		if (hasMaterials)
			grow(triangleMaterials, numTriangles, uint32_t(0));
		largePolygons.resize(numLarge);
		largeCorners.resize(numLargeCorners);
		const Targets targets{positions.data(), colors.empty() ? nullptr : colors.data(), texcoords.data(), normals.data(),
		                      triangles.data(), cornerVT.empty() ? nullptr : cornerVT.data(), cornerVN.empty() ? nullptr : cornerVN.data(),
		                      triangleMaterials.empty() ? nullptr : triangleMaterials.data(), largePolygons.data(), largeCorners.data(), &materialIds};
		ParallelForPool(pool, chunks.size(), [&](size_t i) { ParseChunk(chunks[i], targets); });
		for (const Chunk& chunk : chunks) {
			if (chunk.errorLine != 0) {
				REPORT_WARNING("{}:{}: {}", fileName, chunk.lineBase + chunk.errorLine, chunk.error);
				pending = {};
				vertices.clear();
				return false;
			}
			numSkipped += chunk.numSkipped;
		}

		// the carried tail and the next block form the next round's data
		bytesDone += static_cast<size_t>(end - begin);
		std::vector<char> carry(end, begin + data.size());
		if (pending.valid()) {
			pending.get();
			data.swap(next);
			data.insert(data.begin(), carry.begin(), carry.end());
			if (file)
				pending = std::async(std::launch::async, readBlock, std::ref(next));
		} else {
			data.swap(carry);
		}
	}
	if (triangles.empty()) {
		REPORT_WARNING("invalid mesh file {}: no faces", fileName);
		return false;
	}

	// every index in range, now that the totals are known
	const auto inRange = [](const std::vector<uint32_t>& indices, size_t count, bool optional) {
		return std::all_of(indices.begin(), indices.end(), [&](uint32_t idx) { return (optional && idx == NO_INDEX) || idx < count; });
	};
	bool valid = std::all_of(triangles.begin(), triangles.end(), [&](const Face& face) {
		return face[0] < positions.size() && face[1] < positions.size() && face[2] < positions.size();
	});
	valid = valid && inRange(cornerVT, texcoords.size(), true) && inRange(cornerVN, normals.size(), true);
	valid = valid && std::all_of(largeCorners.begin(), largeCorners.end(), [&](const Corner& corner) {
		        return corner.v < positions.size() && (corner.vt == NO_INDEX || corner.vt < texcoords.size()) && (corner.vn == NO_INDEX || corner.vn < normals.size());
	        });
	if (!valid) {
		REPORT_WARNING("invalid mesh file {}: a face references an element that is not defined", fileName);
		return false;
	}
	if (numSkipped)
		REPORT_WARNING("{}: {} faces with fewer than 3 corners skipped", fileName, numSkipped);

	// triangulate the larger polygons into the slots reserved for them
	{
		ParallelForPool(pool, (largePolygons.size() + 1023) / 1024, [&](size_t block) {
			std::vector<std::array<uint32_t, 3>> polygonTriangles;
			for (size_t i = block * 1024; i < std::min(largePolygons.size(), (block + 1) * 1024); ++i) {
				const LargePolygon& polygon = largePolygons[i];
				const Corner* corners = largeCorners.data() + polygon.firstCorner;
				TriangulatePolygon(positions, corners, polygon.numCorners, polygonTriangles);
				FOREACH (j, polygonTriangles) {
					const size_t t = polygon.firstTriangle + j;
					for (int k = 0; k < 3; ++k) {
						const Corner& corner = corners[polygonTriangles[j][k]];
						triangles[t][k] = corner.v;
						if (!cornerVT.empty())
							cornerVT[t * 3 + k] = corner.vt;
						if (!cornerVN.empty())
							cornerVN[t * 3 + k] = corner.vn;
					}
					if (!triangleMaterials.empty())
						triangleMaterials[t] = polygon.material;
				}
			}
		});
	}
	largePolygons = std::vector<LargePolygon>();
	largeCorners = std::vector<Corner>();

	vertices = std::move(positions);
	faces = std::move(triangles);

	// colors, in 0..1 unless some value says 0..255
	if (numColored == vertices.size()) {
		float maxValue = 0;
		for (const Eigen::Vector3f& color : colors)
			maxValue = std::max(maxValue, color.maxCoeff());
		const float scale = maxValue > 1.f ? 1.f : 255.f;
		vertexColors.resize(vertices.size());
		FOREACH (i, vertexColors) {
			const Eigen::Vector3f c = (colors[i] * scale).array().round().max(0.f).min(255.f);
			vertexColors[i] = Pixel(static_cast<uint8_t>(c[2]), static_cast<uint8_t>(c[1]), static_cast<uint8_t>(c[0]));
		}
	} else if (numColored) {
		REPORT_WARNING("{}: only {} of {} vertices have a color; colors ignored", fileName, numColored, vertices.size());
	}
	colors = std::vector<Eigen::Vector3f>();

	// vertex normals: the one its corners agree on, or the mean of theirs
	if (!cornerVN.empty()) {
		if (std::find(cornerVN.begin(), cornerVN.end(), NO_INDEX) != cornerVN.end()) {
			REPORT_WARNING("{}: not every face corner has a normal; normals ignored", fileName);
		} else {
			std::vector<uint32_t> vertexVN(vertices.size(), NO_INDEX);
			std::vector<bool> mixed(vertices.size(), false);
			FOREACH (c, cornerVN) {
				const VIndex v = faces[c / 3][c % 3];
				const uint32_t vn = cornerVN[c];
				if (vertexVN[v] == NO_INDEX)
					vertexVN[v] = vn;
				else if (vertexVN[v] != vn && normals[vertexVN[v]] != normals[vn])
					mixed[v] = true;
			}
			vertexNormals.resize(vertices.size());
			FOREACH (v, vertexNormals)
				vertexNormals[v] = vertexVN[v] == NO_INDEX ? Normal::Zero() : normals[vertexVN[v]];
			if (std::find(mixed.begin(), mixed.end(), true) != mixed.end()) {
				FOREACH (v, vertexNormals)
					if (mixed[v])
						vertexNormals[v] = Normal::Zero();
				FOREACH (c, cornerVN) {
					const VIndex v = faces[c / 3][c % 3];
					if (mixed[v])
						vertexNormals[v] += normals[cornerVN[c]].normalized();
				}
				FOREACH (v, vertexNormals)
					if (mixed[v])
						vertexNormals[v].normalize();
			}
		}
	}
	normals = std::vector<Normal>();
	cornerVN = std::vector<uint32_t>();

	// materials and texture coordinates
	std::unordered_map<std::string, Material> materials;
	for (const std::string& library : libraries) {
		const std::filesystem::path path = ResolvePath(directory, library);
		std::error_code ec;
		if (!std::filesystem::exists(path, ec) && library.find(' ') != std::string::npos) {
			// several libraries on one line
			for (const char *p = SkipBlanks(library.data(), library.data() + library.size()), *end = library.data() + library.size(); p < end;) {
				const char* tokenEnd = SkipToken(p, end);
				LoadMaterialLibrary(ResolvePath(directory, std::string(p, tokenEnd)), materials);
				p = SkipBlanks(tokenEnd, end);
			}
		} else {
			LoadMaterialLibrary(path, materials);
		}
	}
	// the materials the faces use, in order of first appearance
	std::vector<uint32_t> blobOfMaterial(materialNames.size(), NO_INDEX);
	std::vector<uint32_t> blobMaterials;
	if (triangleMaterials.empty()) {
		blobOfMaterial[0] = 0;
		blobMaterials.push_back(0);
	} else {
		for (const uint32_t m : triangleMaterials) {
			if (blobOfMaterial[m] == NO_INDEX) {
				blobOfMaterial[m] = static_cast<uint32_t>(blobMaterials.size());
				blobMaterials.push_back(m);
			}
		}
	}
	const auto findMaterial = [&](uint32_t m) -> const Material* {
		const auto it = materials.find(materialNames[m]);
		return it == materials.end() ? nullptr : &it->second;
	};
	const bool anyMap = std::any_of(blobMaterials.begin(), blobMaterials.end(), [&](uint32_t m) {
		const Material* mat = findMaterial(m);
		return mat && !mat->map.empty();
	});
	const bool everyCornerVT = !cornerVT.empty() && std::find(cornerVT.begin(), cornerVT.end(), NO_INDEX) == cornerVT.end();
	if (anyMap && blobMaterials.size() > MAX_TEXBLOBS) {
		REPORT_WARNING("{}: {} textured materials exceed the {}-blob limit; texture ignored", fileName, blobMaterials.size(), MAX_TEXBLOBS);
	} else if (anyMap) {
		texturesDiffuse.resize(blobMaterials.size());
		std::vector<std::string> failed(blobMaterials.size());
		ParallelForPool(pool, blobMaterials.size(), [&](size_t b) {
			const Material* mat = findMaterial(blobMaterials[b]);
			if (mat && !mat->map.empty()) {
				try {
					texturesDiffuse[b] = cv::imread(mat->map.string(), cv::IMREAD_COLOR);
				} catch (const std::exception&) {
					texturesDiffuse[b] = Image3u();
				}
				if (texturesDiffuse[b].empty())
					failed[b] = mat->map.string();
			}
			if (texturesDiffuse[b].empty()) {
				// a material without an image keeps its color as a one-texel texture
				const Eigen::Vector3f Kd = (mat ? mat->Kd : Eigen::Vector3f::Ones()) * 255.f;
				const auto channel = [](float value) { return static_cast<uint8_t>(std::clamp(std::lround(value), 0l, 255l)); };
				texturesDiffuse[b] = Image3u(1, 1);
				texturesDiffuse[b](0, 0) = Pixel(channel(Kd[2]), channel(Kd[1]), channel(Kd[0]));
			}
		});
		for (const std::string& name : failed)
			if (!name.empty())
				REPORT_WARNING("{}: missing/unreadable texture '{}'; its material color used instead", fileName, name);
		faceTexcoords.resize(faces.size() * 3);
		FOREACH (c, faceTexcoords)
			faceTexcoords[c] = cornerVT.empty() || cornerVT[c] == NO_INDEX ? TexCoord(0.5f, 0.5f) : texcoords[cornerVT[c]];
		if (blobMaterials.size() > 1) {
			faceTexblobs.resize(faces.size());
			FOREACH (f, faceTexblobs)
				faceTexblobs[f] = static_cast<TexIndex>(blobOfMaterial[triangleMaterials[f]]);
		}
		faceTexcoords = FTexcoordsUnNormalizeFlipY();
	} else if (everyCornerVT) {
		faceTexcoords.resize(faces.size() * 3);
		FOREACH (c, faceTexcoords)
			faceTexcoords[c] = texcoords[cornerVT[c]];
		// the materials still tell the pages of an untextured atlas apart
		if (blobMaterials.size() > 1 && blobMaterials.size() <= MAX_TEXBLOBS) {
			faceTexblobs.resize(faces.size());
			FOREACH (f, faceTexblobs)
				faceTexblobs[f] = static_cast<TexIndex>(blobOfMaterial[triangleMaterials[f]]);
		}
		faceTexcoords = FTexcoordsUnNormalizeFlipY();
	} else if (hasVT) {
		REPORT_WARNING("{}: not every face corner has a texture coordinate and no material has a map; texture coordinates ignored", fileName);
	}
	REPORT_STATUS_NOW("Mesh loaded{}: {}",
	                  texturesDiffuse.empty() ? "" : HALFMESH_FORMAT(" ({} textures)", texturesDiffuse.size()),
	                  fileName);
	return true;
}

// ---------------------------------------------------------------------------
// SaveOBJ
// ---------------------------------------------------------------------------
bool Mesh::SaveOBJ(const std::string& fileName, ImageFormat imageFormat) const
{
	SyncFacesConst();
	const std::filesystem::path path(fileName);
	const std::string stem(path.stem().string());
	const std::filesystem::path directory(path.parent_path());
	const bool hasTexcoords = !faces.empty() && faceTexcoords.size() == faces.size() * 3;
	const bool textured = hasTexcoords && !texturesDiffuse.empty();
	ASSERT(faceTexblobs.empty() || faceTexblobs.size() == faces.size());
	ASSERT(texturesDiffuse.size() <= MAX_TEXBLOBS);
	BS::light_thread_pool pool;

	// the material library and its textures
	if (textured) {
		const char* extension = imageFormat == ImageFormat::PNG ? "png" : "jpg";
		std::vector<std::string> textureNames(texturesDiffuse.size());
		FOREACH (i, textureNames)
			textureNames[i] = HALFMESH_FORMAT("{}_material_{:02}_map_Kd.{}", stem, i, extension);
		// one byte per texture, not std::vector<bool>: the workers write their flags concurrently
		std::vector<uint8_t> written(texturesDiffuse.size(), 0);
		const std::vector<int> codecParams{cv::IMWRITE_JPEG_QUALITY, 95};
		ParallelForPool(pool, texturesDiffuse.size(), [&](size_t i) {
			try {
				written[i] = !texturesDiffuse[i].empty() && cv::imwrite((directory / textureNames[i]).string(), texturesDiffuse[i], codecParams);
			} catch (const std::exception&) {
				written[i] = false;
			}
		});
		std::ofstream mtl(directory / (stem + ".mtl"), std::ios::binary);
		mtl << "# halfmesh " << Version() << "\n";
		FOREACH (i, texturesDiffuse) {
			mtl << HALFMESH_FORMAT("newmtl material_{:02}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nNs 1\n", i);
			if (written[i])
				mtl << "map_Kd " << textureNames[i] << "\n";
			else
				REPORT_WARNING("failed to write texture '{}' for {}; material saved without it", textureNames[i], fileName);
		}
		if (!mtl.flush()) {
			REPORT_WARNING("failed to write the material library of {}", fileName);
			return false;
		}
	}

	// one vt per distinct UV of a vertex: bucket the corners by vertex, sort each
	// bucket by UV, and number the groups in order of first use
	std::vector<TexCoord> uvs;
	std::vector<uint32_t> cornerVT;
	std::vector<uint32_t> vtCorners; // the corner holding each vt's value
	std::vector<FIndex> order; // the faces grouped by texture blob
	// blobs without textures (the pages of an untextured atlas) still group
	const size_t numGroups = !hasTexcoords || faceTexblobs.empty() ? 1 : textured ? texturesDiffuse.size()
	                                                                              : size_t(*std::max_element(faceTexblobs.begin(), faceTexblobs.end())) + 1;
	if (numGroups > 1) {
		std::vector<size_t> offsets(numGroups + 1, 0);
		for (const TexIndex idxTexblob : faceTexblobs) {
			ASSERT(idxTexblob < numGroups);
			++offsets[idxTexblob + 1];
		}
		std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
		order.resize(faces.size());
		FOREACH (f, faceTexblobs)
			order[offsets[faceTexblobs[f]]++] = static_cast<FIndex>(f);
	}
	const auto faceAt = [&](size_t i) -> FIndex { return order.empty() ? static_cast<FIndex>(i) : order[i]; };
	if (hasTexcoords) {
		uvs = FTexcoordsNormalizeFlipY();
		const size_t numCorners = uvs.size();
		std::vector<size_t> offsets(vertices.size() + 2, 0);
		for (const Face& face : faces)
			for (int k = 0; k < 3; ++k)
				++offsets[face[k] + 2];
		std::partial_sum(offsets.begin() + 2, offsets.end(), offsets.begin() + 2);
		std::vector<uint32_t> buckets(numCorners);
		FOREACH (f, faces)
			for (int k = 0; k < 3; ++k)
				buckets[offsets[faces[f][k] + 1]++] = static_cast<uint32_t>(f * 3 + k);
		// the first corner of each group of bit-identical UVs, per corner
		const auto uvBits = [&](uint32_t c) {
			uint64_t bits;
			static_assert(sizeof(TexCoord) == sizeof(bits));
			std::memcpy(&bits, uvs[c].data(), sizeof(bits));
			return bits;
		};
		std::vector<uint32_t> groupOf(numCorners);
		ParallelForPool(pool, (vertices.size() + 4095) / 4096, [&](size_t block) {
			for (size_t v = block * 4096; v < std::min(vertices.size(), (block + 1) * 4096); ++v) {
				uint32_t* const first = buckets.data() + offsets[v];
				uint32_t* const last = buckets.data() + offsets[v + 1];
				std::sort(first, last, [&](uint32_t a, uint32_t b) { return std::make_pair(uvBits(a), a) < std::make_pair(uvBits(b), b); });
				for (const uint32_t* c = first; c != last; ++c)
					groupOf[*c] = c != first && uvBits(*c) == uvBits(c[-1]) ? groupOf[c[-1]] : *c;
			}
		});
		cornerVT.assign(numCorners, NO_INDEX);
		std::vector<uint32_t> vtOfGroup(numCorners, NO_INDEX);
		for (size_t i = 0; i < faces.size(); ++i) {
			const FIndex f = faceAt(i);
			for (int k = 0; k < 3; ++k) {
				const uint32_t c = static_cast<uint32_t>(f * 3 + k);
				uint32_t& vt = vtOfGroup[groupOf[c]];
				if (vt == NO_INDEX) {
					vt = static_cast<uint32_t>(vtCorners.size());
					vtCorners.push_back(c);
				}
				cornerVT[c] = vt;
			}
		}
	}
	// authored normals win; otherwise derive them from the face normals, as SavePLY does
	std::vector<Normal> derivedNormals;
	if (vertexNormals.empty() && !faceNormals.empty())
		derivedNormals = const_cast<Mesh&>(*this).ComputeVertexNormals();
	const std::vector<Normal>& normals = vertexNormals.empty() ? derivedNormals : vertexNormals;
	const bool hasNormals = !normals.empty();

	std::ofstream out(fileName, std::ios::binary);
	if (!out) {
		REPORT_WARNING("Could not open file: {}", fileName);
		return false;
	}
	out << "# halfmesh " << Version() << ": " << vertices.size() << " vertices, " << faces.size() << " faces\n";
	if (textured)
		out << "mtllib " << stem << ".mtl\n";
	WriteLines(out, pool, vertices.size(), [&](std::string& text, size_t v) {
		text += "v ";
		AppendFloat(text, vertices[v].x());
		text += ' ';
		AppendFloat(text, vertices[v].y());
		text += ' ';
		AppendFloat(text, vertices[v].z());
		if (!vertexColors.empty()) {
			const Pixel& color = vertexColors[v];
			for (int channel = 2; channel >= 0; --channel) {
				text += ' ';
				AppendFloat(text, static_cast<float>(color[channel]) / 255.f);
			}
		}
		text += '\n';
	});
	WriteLines(out, pool, vtCorners.size(), [&](std::string& text, size_t vt) {
		const TexCoord& uv = uvs[vtCorners[vt]];
		text += "vt ";
		AppendFloat(text, uv.x());
		text += ' ';
		AppendFloat(text, uv.y());
		text += '\n';
	});
	if (hasNormals) {
		WriteLines(out, pool, normals.size(), [&](std::string& text, size_t v) {
			text += "vn ";
			AppendFloat(text, normals[v].x());
			text += ' ';
			AppendFloat(text, normals[v].y());
			text += ' ';
			AppendFloat(text, normals[v].z());
			text += '\n';
		});
	}
	const auto writeFaces = [&](size_t begin, size_t end) {
		WriteLines(out, pool, end - begin, [&](std::string& text, size_t i) {
			const FIndex f = faceAt(begin + i);
			text += 'f';
			for (int k = 0; k < 3; ++k) {
				text += ' ';
				AppendIndex(text, faces[f][k]);
				if (hasTexcoords || hasNormals) {
					text += '/';
					if (hasTexcoords)
						AppendIndex(text, cornerVT[f * 3 + k]);
					if (hasNormals) {
						text += '/';
						AppendIndex(text, faces[f][k]);
					}
				}
			}
			text += '\n';
		});
	};
	if (!textured && numGroups == 1) {
		writeFaces(0, faces.size());
	} else {
		size_t begin = 0;
		for (size_t i = 0; i < numGroups; ++i) {
			size_t end = begin;
			while (end < faces.size() && (faceTexblobs.empty() ? TexIndex(0) : faceTexblobs[faceAt(end)]) == i)
				++end;
			if (end == begin)
				continue;
			out << HALFMESH_FORMAT("usemtl material_{:02}\n", i);
			writeFaces(begin, end);
			begin = end;
		}
		ASSERT(begin == faces.size());
	}
	if (!out.flush()) {
		REPORT_WARNING("failed to write {}", fileName);
		return false;
	}
	return true;
}

} // namespace halfmesh
