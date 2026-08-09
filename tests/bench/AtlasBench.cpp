/*
* AtlasBench.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/AtlasBench.cpp — UV-atlas benchmark CLI (atlasbench).
//
// Loads a mesh, runs halfmesh's atlas pipeline plus the selected SOTA baseline
// engines on it, measures each with the shared metrics, and writes a
// per-stage comparison report (report.{json,md,csv}).
//
//   atlasbench --mesh <p> [--tier small|medium|large] [--resolution N]
//              [--engines halfmesh,xatlas,...] [--stage all|segment|param|pack]
//              [--decimate N] [--timeout-s S] [--out <dir>] [--seed N]
#include "BenchTypes.h"
#include "BenchReport.h"
#include "BenchMetrics.h" // hmbench::PeakRssBytes
#include "engines/EngineHalfmesh.h"
#if defined(HMBENCH_WITH_XATLAS)
	#include "engines/EngineXatlas.h"
#endif
#if defined(HMBENCH_WITH_PMP)
	#include "engines/EnginePmp.h"
#endif
#if defined(HMBENCH_WITH_LIBIGL)
	#include "engines/EngineLibigl.h"
#endif
#if defined(HMBENCH_WITH_CGAL)
	#include "engines/EngineCgal.h"
#endif
#if defined(HMBENCH_WITH_BFF)
	#include "engines/EngineBff.h"
#endif

#include <halfmesh/Mesh.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/Util/Hash.h> // std::hash<std::tuple<...>>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

std::vector<std::string> Split(const std::string& s, char sep)
{
	std::vector<std::string> out;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, sep))
		if (!item.empty())
			out.push_back(item);
	return out;
}

void Usage()
{
	std::cerr << "Usage: atlasbench --mesh <path> [options]\n"
	             "  --mesh <path>          input mesh (.ply/.obj/.glb)\n"
	             "  --tier small|medium|large   (default small)\n"
	             "  --resolution N         atlas side length in texels (default 1024)\n"
	             "  --engines a,b,c        engines to run (default: halfmesh)\n"
	             "  --stage all|segment|param|pack  (default all)\n"
	             "  --decimate N           decimate to ~N faces before benchmarking (0=off)\n"
	             "  --timeout-s S          per-stage soft timeout seconds (default 120)\n"
	             "  --out <dir>            output directory for report.{json,md,csv} (default .)\n"
	             "  --seed N               RNG seed (default 42)\n";
}

} // namespace

int main(int argc, char* argv[])
{
	hmbench::BenchConfig cfg;
	cfg.engines = {
	    "halfmesh",
#if defined(HMBENCH_WITH_XATLAS)
	    "xatlas",
#endif
	};

	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto next = [&](const char* name) -> std::string {
			if (i + 1 >= argc) {
				std::cerr << "error: " << name << " requires a value\n";
				std::exit(EXIT_FAILURE);
			}
			return argv[++i];
		};
		if (a == "--mesh")
			cfg.meshPath = next("--mesh");
		else if (a == "--tier")
			cfg.tier = next("--tier");
		else if (a == "--resolution")
			cfg.resolution = std::stoul(next("--resolution"));
		else if (a == "--engines")
			cfg.engines = Split(next("--engines"), ',');
		else if (a == "--stage")
			cfg.stage = next("--stage");
		else if (a == "--decimate")
			cfg.decimateTargetFaces = std::stoul(next("--decimate"));
		else if (a == "--timeout-s")
			cfg.perStageTimeoutS = std::stod(next("--timeout-s"));
		else if (a == "--out")
			cfg.outDir = next("--out");
		else if (a == "--seed")
			cfg.seed = std::stoul(next("--seed"));
		else if (a == "--flatten-iters")
			cfg.flattenIterations = std::stoi(next("--flatten-iters"));
		else if (a == "--init")
			cfg.initMethod = next("--init");
		else if (a == "--orient")
			cfg.orient = next("--orient");
		else if (a == "--save-mesh")
			cfg.saveMesh = next("--save-mesh");
		else if (a == "--repair")
			cfg.repair = true;
		else if (a == "--fix-manifold")
			cfg.fixManifold = true;
		else if (a == "--diagnose")
			cfg.diagnose = true;
		else if (a == "--dev-smooth")
			cfg.devSmooth = std::stoi(next("--dev-smooth"));
		else if (a == "--flip-rounds")
			cfg.flipRounds = std::stoi(next("--flip-rounds"));
		else if (a == "--seed-mult")
			cfg.seedMult = std::stof(next("--seed-mult"));
		else if (a == "--cone-err")
			cfg.coneErr = std::stof(next("--cone-err"));
		else if (a == "--vdefect")
			cfg.vdefect = std::stof(next("--vdefect"));
		else if (a == "--max-distortion")
			cfg.maxDistortion = std::stof(next("--max-distortion"));
		else if (a == "--cut-to-disk")
			cfg.cutToDisk = true;
		else if (a == "-h" || a == "--help") {
			Usage();
			return EXIT_SUCCESS;
		} else {
			std::cerr << "error: unknown argument '" << a << "'\n";
			Usage();
			return EXIT_FAILURE;
		}
	}

	if (cfg.meshPath.empty()) {
		std::cerr << "error: --mesh is required\n";
		Usage();
		return EXIT_FAILURE;
	}

	halfmesh::Mesh mesh;
	if (!mesh.Load(cfg.meshPath)) {
		std::cerr << "error: failed to load '" << cfg.meshPath << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "Loaded " << mesh.vertices.size() << " vertices, "
	          << mesh.faces.size() << " faces from " << cfg.meshPath << "\n";

	if (cfg.repair) {
		// Clean dirty / non-manifold input (common in photogrammetry meshes)
		// so the half-edge-based pipeline does not choke. Canonical sequence.
		// Weld first: glTF stores textured meshes unwelded (one disconnected
		// sub-mesh per texture), so welding coincident vertices reconnects them.
		mesh.RemoveDuplicateVertices();
		mesh.RemoveDuplicateFaces();
		mesh.RemoveDegenerateFaces();
		mesh.RemoveUnreferencedVertices();
		mesh.FixNonManifold();
		mesh.RemoveUnreferencedVertices();
		std::cout << "Repaired -> " << mesh.vertices.size() << " vertices, "
		          << mesh.faces.size() << " faces\n";
	}

	if (cfg.fixManifold) {
		// Minimal manifold repair: split non-manifold vertices/edges only (keeps
		// every face — unlike --repair, which also strips degenerate/duplicate
		// faces). FixNonManifold duplicates vertices, so drop the now-unreferenced.
		const std::size_t f0 = mesh.faces.size();
		mesh.FixNonManifold();
		mesh.RemoveUnreferencedVertices();
		std::cout << "FixNonManifold -> " << mesh.vertices.size() << " vertices, "
		          << mesh.faces.size() << " faces (was " << f0 << ")\n";
	}

	if (cfg.diagnose) {
		// Mesh-health probe: cheap O(F) checks (no half-edge build) so we can
		// safely characterize a mesh that crashes/hangs the manifold-only
		// half-edge pipeline.  Reports degeneracy, coincident vertices and the
		// exact non-manifold conditions HalfMesh::Build assumes away.
		std::size_t degenerate = 0;
		for (std::size_t f = 0; f < mesh.faces.size(); ++f)
			if (mesh.ComputeFaceDoubleArea(static_cast<halfmesh::Mesh::FIndex>(f)) <= 0.f)
				++degenerate;

		// coincident vertices (bit-exact position match)
		std::unordered_map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> posSeen;
		posSeen.reserve(mesh.vertices.size() * 2);
		std::size_t coincident = 0;
		for (const halfmesh::Mesh::Vertex& v : mesh.vertices) {
			const std::tuple<uint32_t, uint32_t, uint32_t> key{
			    std::bit_cast<uint32_t>(v.x()), std::bit_cast<uint32_t>(v.y()),
			    std::bit_cast<uint32_t>(v.z())};
			if (!posSeen.emplace(key, 0u).second)
				++coincident;
		}

		// directed-edge collisions (the Build assert: a directed edge seen twice)
		// and undirected-edge valence (an undirected edge in >2 faces => non-manifold)
		std::unordered_map<uint64_t, uint32_t> directed, undirected;
		directed.reserve(mesh.faces.size() * 3);
		undirected.reserve(mesh.faces.size() * 3);
		auto key2 = [](uint32_t a, uint32_t b) { return (static_cast<uint64_t>(a) << 32) | b; };
		std::size_t selfEdges = 0;
		for (const halfmesh::Mesh::Face& face : mesh.faces) {
			for (int e = 0; e < 3; ++e) {
				const uint32_t a = face[e], b = face[(e + 1) % 3];
				if (a == b) {
					++selfEdges;
					continue;
				}
				++directed[key2(a, b)];
				++undirected[key2(std::min(a, b), std::max(a, b))];
			}
		}
		std::size_t directedCollisions = 0;
		for (const auto& [k, c] : directed)
			if (c > 1)
				++directedCollisions;
		std::size_t nonmanifoldEdges = 0, boundaryEdges = 0;
		for (const auto& [k, c] : undirected) {
			if (c > 2)
				++nonmanifoldEdges;
			else if (c == 1)
				++boundaryEdges;
		}

		std::cout << "=== mesh diagnosis ===\n"
		          << "vertices              : " << mesh.vertices.size() << "\n"
		          << "faces                 : " << mesh.faces.size() << "\n"
		          << "degenerate faces      : " << degenerate << " ("
		          << (mesh.faces.empty() ? 0.0 : 100.0 * static_cast<double>(degenerate) / static_cast<double>(mesh.faces.size())) << "%)\n"
		          << "coincident vertices   : " << coincident << "\n"
		          << "self-edges (a==b)     : " << selfEdges << "\n"
		          << "directed-edge dups    : " << directedCollisions
		          << "  (HalfMesh::Build assumes 0)\n"
		          << "non-manifold edges>2f : " << nonmanifoldEdges
		          << "  (HalfMesh::Build assumes 0)\n"
		          << "boundary edges (1f)   : " << boundaryEdges << "\n"
		          << "manifold?             : "
		          << ((directedCollisions == 0 && nonmanifoldEdges == 0 && selfEdges == 0) ? "YES" : "NO")
		          << "\n"
		          << "peak RSS (MB)         : " << (hmbench::PeakRssBytes() / (std::size_t{1024} * 1024)) << "\n";
		return EXIT_SUCCESS;
	}

	if (cfg.decimateTargetFaces > 0 && mesh.faces.size() > cfg.decimateTargetFaces) {
		const float ratio = static_cast<float>(cfg.decimateTargetFaces) / static_cast<float>(mesh.faces.size());
		mesh.Simplify(ratio);
		std::cout << "Decimated to " << mesh.faces.size() << " faces (ratio "
		          << ratio << ")\n";
	}

	if (!cfg.saveMesh.empty()) {
		if (!mesh.Save(cfg.saveMesh)) {
			std::cerr << "error: failed to save '" << cfg.saveMesh << "'\n";
			return EXIT_FAILURE;
		}
		std::cout << "Saved (decimated) mesh to " << cfg.saveMesh << " ("
		          << mesh.faces.size() << " faces)\n";
		return EXIT_SUCCESS;
	}

	std::vector<hmbench::EngineResult> results;

	if (cfg.stage == "param") {
		// Parametrization isolation: one shared chart partition (halfmesh's
		// SegmentCharts), flattened by every selected engine.  Default engine
		// set = halfmesh + all compiled per-chart baselines.
		if (cfg.engines == std::vector<std::string>{"halfmesh"
#if defined(HMBENCH_WITH_XATLAS)
		                                            ,
		                                            "xatlas"
#endif
		    }) {
			cfg.engines = {"halfmesh"};
#if defined(HMBENCH_WITH_LIBIGL)
			cfg.engines.push_back("libigl-lscm");
#endif
#if defined(HMBENCH_WITH_PMP)
			cfg.engines.push_back("pmp-lscm");
			cfg.engines.push_back("pmp-harmonic");
#endif
#if defined(HMBENCH_WITH_CGAL)
			cfg.engines.push_back("cgal-lscm");
			cfg.engines.push_back("cgal-arap");
#endif
#if defined(HMBENCH_WITH_BFF)
			cfg.engines.push_back("bff");
#endif
		}
		halfmesh::Mesh seg = mesh;
		halfmesh::ParametrizeParams pp;
		std::vector<unsigned> faceChart;
		const unsigned nc = halfmesh::SegmentCharts(seg, pp, faceChart);
		std::cout << "Shared segmentation: " << nc << " charts\n";
		for (const std::string& engine : cfg.engines) {
			std::cout << "Param engine: " << engine << " ..." << std::flush;
			if (engine == "halfmesh") {
				results.push_back(hmbench::RunHalfmeshParam(seg, faceChart, nc, cfg));
#if defined(HMBENCH_WITH_LIBIGL)
			} else if (engine == "libigl-lscm") {
				results.push_back(hmbench::RunLibiglLscm(seg, faceChart, nc, cfg));
#endif
#if defined(HMBENCH_WITH_PMP)
			} else if (engine == "pmp-lscm") {
				results.push_back(hmbench::RunPmpLscm(seg, faceChart, nc, cfg));
			} else if (engine == "pmp-harmonic") {
				results.push_back(hmbench::RunPmpHarmonic(seg, faceChart, nc, cfg));
#endif
#if defined(HMBENCH_WITH_CGAL)
			} else if (engine == "cgal-lscm") {
				results.push_back(hmbench::RunCgalLscm(seg, faceChart, nc, cfg));
			} else if (engine == "cgal-arap") {
				results.push_back(hmbench::RunCgalArap(seg, faceChart, nc, cfg));
#endif
#if defined(HMBENCH_WITH_BFF)
			} else if (engine == "bff") {
				results.push_back(hmbench::RunBffParam(seg, faceChart, nc, cfg));
#endif
			} else {
				hmbench::EngineResult r;
				r.engine = engine;
				r.note = "engine not available for --stage param in this build";
				results.push_back(r);
			}
			std::cout << " done\n";
		}
	} else {
		for (const std::string& engine : cfg.engines) {
			std::cout << "Running engine: " << engine << " ..." << std::flush;
			if (engine == "halfmesh") {
				results.push_back(hmbench::RunHalfmesh(mesh, cfg));
#if defined(HMBENCH_WITH_XATLAS)
			} else if (engine == "xatlas") {
				results.push_back(hmbench::RunXatlas(mesh, cfg));
#endif
#if defined(HMBENCH_WITH_CGAL)
			} else if (engine == "cgal-sdf") {
				results.push_back(hmbench::RunCgalSegment(mesh, cfg));
#endif
			} else {
				hmbench::EngineResult r;
				r.engine = engine;
				r.note = "engine not available in this build";
				results.push_back(r);
			}
			std::cout << " done\n";
		}
	}

	const std::string report = hmbench::RenderMarkdown(cfg, results);
	std::cout << "\n"
	          << report;

	if (!hmbench::WriteReports(cfg, results)) {
		std::cerr << "warning: failed to write report files to '" << cfg.outDir << "'\n";
	} else {
		std::cout << "Wrote report.{json,md,csv} to " << cfg.outDir << "\n";
	}
	return EXIT_SUCCESS;
}
