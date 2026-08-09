/*
* RemeshBench.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/RemeshBench.cpp — isotropic-remeshing quality + speed benchmark.
//
// Runs halfmesh's Mesh::RemeshIsotropic over a corpus of meshes at several target
// edge lengths and reports, per run: wall-time and the quality scalars an
// isotropic remesher is judged on (edge-length size uniformity, triangle-angle
// quality, vertex valence regularity, fidelity to the input surface) — all via
// the shared hmtest::metrics toolkit.
//
// Optional baseline (opt-in at configure time, mirroring atlasbench):
//   -DHALFMESH_REMESH_WITH_PMP=ON   compares against pmp::uniform_remeshing
// It runs on the same inputs/parameters so quality and time are directly comparable.
//
// Build:  cmake -DHALFMESH_BUILD_BENCH=ON [-DHALFMESH_REMESH_WITH_PMP=ON] ...
// Run:    ./tests/bench/remeshbench            (default corpus + sweep)
//         ./tests/bench/remeshbench --mesh path.ply
// Output: a stdout table + tests/bench/results/remesh_baseline.json
//
// Exit code is non-zero if any run produced an empty mesh or a non-finite metric,
// so the binary doubles as a CI smoke test (registered with ctest, label "bench").

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include "metrics/Metrics.h"
#include "metrics/RemeshQuality.h"
#include "corpus/Corpus.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// Optional baseline engine (defined in remesh_engines/*.cpp, compiled only when
// the matching option is ON).  Declared at global scope to match its linkage.
#if HMBENCH_WITH_PMP
bool RemeshPmp(const halfmesh::Mesh& in, float edgeLen, int iters,
               halfmesh::Mesh& out, double& seconds);
#endif

namespace {

using Clock = std::chrono::steady_clock;
static double elapsedS(Clock::time_point t0)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - t0).count();
}

// ---- engine adapters -------------------------------------------------------
// Each runs an isotropic remesh of `in` to target `edgeLen` for `iters`
// iterations, writing the result to `out` and the remesh-only wall-time to
// `seconds`.  Returns false if the engine failed/unavailable.

static bool RemeshHalfmesh(const halfmesh::Mesh& in, float edgeLen, int iters,
                           halfmesh::Mesh& out, double& seconds)
{
	out = in;
	halfmesh::Mesh::RemeshParams p;
	p.SetEdgeLength(edgeLen);
	p.iterations = iters;
	halfmesh::Mesh::RemeshStats s;
	const auto t0 = Clock::now();
	out.RemeshIsotropic(p, &s);
	seconds = elapsedS(t0);
	// Per-pass breakdown to stderr (profiling the speedup work).
	std::fprintf(stderr,
	             "    [pass-s] split=%.3f collapse=%.3f tag=%.3f flip=%.3f smooth=%.3f project=%.3f"
	             "  (nsplit=%u ncollapse=%u nflip=%u)\n",
	             s.splitSeconds, s.collapseSeconds, s.tagSeconds, s.flipSeconds,
	             s.smoothSeconds, s.projectSeconds, s.splitCount, s.collapseCount, s.flipCount);
	return !out.faces.empty();
}

// ---- one benchmark row -----------------------------------------------------
struct Row
{
	std::string engine;
	std::string mesh;
	double factor = 0; // target edge length as a multiple of the input mean
	float edgeLen = 0;
	double seconds = 0;
	hmtest::metrics::RemeshQuality q;
	bool ok = false;
};

static bool finiteQuality(const hmtest::metrics::RemeshQuality& q)
{
	return std::isfinite(q.edgeCov) && std::isfinite(q.minAngleMeanDeg)
	       && std::isfinite(q.meanValence) && std::isfinite(q.meanDistToInput)
	       && q.numFaces > 0;
}

static double InputMeanEdge(const halfmesh::Mesh& in)
{
	halfmesh::Mesh tmp = in;
	tmp.ListHalfEdges();
	return hmtest::metrics::ComputeEdgeLengthStats(tmp).meanLen;
}

static void PrintHeader()
{
	std::printf("%-9s %-12s %6s %10s %9s %7s %8s %8s %8s %8s %10s %10s\n",
	            "engine", "mesh", "factor", "edge_len", "time_s", "Fout",
	            "edge_cov", "minang", "frac<30", "meanval", "dist/bbox", "haus/bbox");
}

static void PrintRow(const Row& r)
{
	std::printf("%-9s %-12s %6.2f %10.5f %9.4f %7zu %8.4f %8.2f %8.4f %8.3f %10.2e %10.2e\n",
	            r.engine.c_str(), r.mesh.c_str(), r.factor, r.edgeLen, r.seconds,
	            r.q.numFaces, r.q.edgeCov, r.q.minAngleMeanDeg, r.q.fracMinAngleLt30,
	            r.q.meanValence, r.q.meanDistRatio(), r.q.hausdorffRatio());
}

static void WriteJson(const std::vector<Row>& rows, const std::string& path)
{
	std::error_code ec;
	std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
	std::ofstream f(path);
	if (!f) {
		std::cerr << "Cannot write " << path << "\n";
		return;
	}
	f << "{\n  \"note\": \"absolute time is machine-dependent; compare quality across engines/params\",\n";
	f << "  \"runs\": [\n";
	for (size_t i = 0; i < rows.size(); ++i) {
		const Row& r = rows[i];
		f << "    {\"engine\": \"" << r.engine << "\""
		  << ", \"mesh\": \"" << r.mesh << "\""
		  << ", \"factor\": " << r.factor
		  << ", \"edge_len\": " << r.edgeLen
		  << ", \"seconds\": " << r.seconds
		  << ", \"faces\": " << r.q.numFaces
		  << ", \"vertices\": " << r.q.numVertices
		  << ", \"edge_cov\": " << r.q.edgeCov
		  << ", \"min_angle_mean_deg\": " << r.q.minAngleMeanDeg
		  << ", \"frac_min_angle_lt_30\": " << r.q.fracMinAngleLt30
		  << ", \"mean_valence\": " << r.q.meanValence
		  << ", \"frac_irregular_valence\": " << r.q.fracIrregularValence
		  << ", \"mean_dist_ratio\": " << r.q.meanDistRatio()
		  << ", \"hausdorff_ratio\": " << r.q.hausdorffRatio()
		  << "}";
		if (i + 1 < rows.size())
			f << ",";
		f << "\n";
	}
	f << "  ]\n}\n";
	std::cout << "\nWrote " << path << " (" << rows.size() << " runs)\n";
}

// A benchmark input materialized lazily so only one mesh is resident at a time
// (the tests/data photogrammetry meshes are hundreds of MB / millions of faces).
struct Source
{
	std::string name;
	std::function<halfmesh::Mesh()> make;
};

// Above this, skip refinement factors (which would multiply the face count) and
// use fewer iterations, to keep wall-time / memory bounded.
constexpr size_t LARGE_FACES = 400000;
// pmp's incremental SurfaceMesh is memory-heavy; skip it above this to avoid OOM
// on the largest meshes (halfmesh still runs).
constexpr size_t PMP_MAX_FACES = 4000000;

} // namespace

int main(int argc, char** argv)
{
	namespace hm = hmtest::metrics;

	bool includeData = false; // --data: also bench every mesh in tests/data
	std::vector<std::string> extraPaths;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a == "--data")
			includeData = true;
		else if (a == "--mesh" && i + 1 < argc)
			extraPaths.push_back(argv[++i]);
	}

	const std::filesystem::path dataDir =
	    std::filesystem::path(__FILE__).parent_path().parent_path() / "data";

	// Controlled corpus (always; keeps the no-arg ctest smoke run fast) + the
	// small bundled real mesh.
	std::vector<Source> sources = {
	    {"uvsphere", [] { return hmtest::corpus::UVSphere(32, 48); }},
	    {"torus", [] { return hmtest::corpus::TorusMesh(48, 24); }},
	    {"gridplane", [] { return hmtest::corpus::GridPlane(40); }},
	    {"cone", [] { return hmtest::corpus::Cone(64); }},
	};
	auto addFile = [&](const std::filesystem::path& p) {
		sources.push_back({p.stem().string(), [p] {
			                   halfmesh::Mesh m;
			                   if (!m.Load(p.string()) || m.faces.empty())
				                   return m;
			                   // Repair raw photogrammetry meshes so they meet
			                   // RemeshIsotropic's manifold / no-degenerate-face
			                   // precondition (zero-area faces are common).
			                   m.RemoveDuplicateVertices();
			                   m.RemoveDuplicateFaces();
			                   m.RemoveDegenerateFaces(0.f);
			                   m.RemoveUnreferencedVertices();
			                   m.FixNonManifold();
			                   return m;
		                   }});
	};
	if (std::filesystem::exists(dataDir / "mesh.ply"))
		addFile(dataDir / "mesh.ply");
	// --data: every top-level mesh file in tests/data (the heavy real meshes);
	// the golden/ fixtures subdir is intentionally excluded.
	if (includeData && std::filesystem::is_directory(dataDir)) {
		std::vector<std::filesystem::path> files;
		for (const auto& e : std::filesystem::directory_iterator(dataDir)) {
			if (!e.is_regular_file())
				continue;
			std::string ext = e.path().extension().string();
			for (char& c : ext)
				c = static_cast<char>(std::tolower(c));
			if ((ext == ".ply" || ext == ".glb" || ext == ".gltf" || ext == ".obj") && e.path().filename() != "mesh.ply")
				files.push_back(e.path());
		}
		std::sort(files.begin(), files.end()); // deterministic order
		for (const auto& p : files)
			addFile(p);
	}
	for (const std::string& p : extraPaths)
		addFile(p);

	std::vector<Row> rows;
	PrintHeader();
	bool allOk = true;

	for (const Source& src : sources) {
		const halfmesh::Mesh in = src.make();
		if (in.faces.empty()) {
			std::cerr << "[skip] " << src.name << ": empty / failed to load\n";
			continue;
		}
		const size_t nfIn = in.faces.size();
		const double meanEdge = InputMeanEdge(in);
		if (!(meanEdge > 0.0)) {
			std::cerr << "[skip] " << src.name << ": degenerate input edges\n";
			continue;
		}
		const bool large = nfIn > LARGE_FACES;
		const int iters = large ? 3 : 5;
		const std::vector<double> factors = large ? std::vector<double>{1.0}
		                                          : std::vector<double>{0.5, 1.0, 2.0};
		std::cerr << "[mesh] " << src.name << " faces=" << nfIn
		          << (large ? " (large: factor 1.0 only)" : "") << "\n";

		for (double factor : factors) {
			const float edgeLen = static_cast<float>(meanEdge * factor);

			auto run = [&](const char* engine,
			               bool (*fn)(const halfmesh::Mesh&, float, int, halfmesh::Mesh&, double&)) {
				Row r;
				r.engine = engine;
				r.mesh = src.name;
				r.factor = factor;
				r.edgeLen = edgeLen;
				halfmesh::Mesh out;
				r.ok = fn(in, edgeLen, iters, out, r.seconds);
				if (r.ok) {
					r.q = hmtest::metrics::ComputeRemeshQuality(in, out);
					r.ok = finiteQuality(r.q);
				}
				if (!r.ok)
					allOk = false;
				PrintRow(r);
				rows.push_back(std::move(r));
			};

			run("halfmesh", &RemeshHalfmesh);
#if HMBENCH_WITH_PMP
			if (nfIn <= PMP_MAX_FACES)
				run("pmp", &RemeshPmp);
			else
				std::cerr << "[skip] pmp on " << src.name << " (" << nfIn << " faces > cap)\n";
#endif
		}
	}

	const std::string outPath = (std::filesystem::path(__FILE__).parent_path()
	                             / "results" / "remesh_baseline.json")
	                                .string();
	WriteJson(rows, outPath);

	if (!allOk) {
		std::cerr << "\nFAIL: one or more runs produced an empty mesh or non-finite metric\n";
		return 1;
	}
	std::cout << "\nOK: " << rows.size() << " runs, all metrics finite\n";
	return 0;
}
