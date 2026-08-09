/*
* BenchReport.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "BenchReport.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace hmbench {

namespace {

inline bool IsSet(double v) { return v != UNSET && std::isfinite(v); }

std::string Num(double v, int prec = 4)
{
	if (!IsSet(v))
		return "—";
	std::ostringstream os;
	os << std::fixed << std::setprecision(prec) << v;
	return os.str();
}

std::string Pct(double v) // v in [0,1]
{
	if (!IsSet(v))
		return "—";
	std::ostringstream os;
	os << std::fixed << std::setprecision(1) << (v * 100.0) << "%";
	return os.str();
}

std::string IntOrDash(int v) { return (v < 0) ? "—" : std::to_string(v); }

std::string Secs(double s)
{
	std::ostringstream os;
	os << std::fixed << std::setprecision(3) << s;
	return os.str();
}

// JSON string escape (minimal — engine names / paths only).
std::string J(const std::string& s)
{
	std::string o = "\"";
	for (char ch : s) {
		if (ch == '"' || ch == '\\')
			o += '\\';
		o += ch;
	}
	o += "\"";
	return o;
}

std::string JNum(double v)
{
	if (!IsSet(v))
		return "null";
	std::ostringstream os;
	os << std::setprecision(9) << v;
	return os.str();
}

} // namespace

std::string RenderMarkdown(const BenchConfig& cfg,
                           const std::vector<EngineResult>& results)
{
	std::ostringstream md;
	md << "# atlasbench report\n\n";
	md << "- **mesh**: `" << cfg.meshPath << "`\n";
	md << "- **tier**: " << cfg.tier << " | **resolution**: " << cfg.resolution
	   << " | **stage**: " << cfg.stage << "\n";
	md << "- **engines**: ";
	for (size_t i = 0; i < results.size(); ++i)
		md << (i ? ", " : "") << results[i].engine;
	md << "\n\n";

	// --- Segmentation -------------------------------------------------------
	md << "## Segmentation\n\n";
	md << "| engine | charts | seam len | planarity (rad) | compactness | coverage | time (s) |\n";
	md << "|---|--:|--:|--:|--:|:--:|--:|\n";
	for (const auto& r : results) {
		const auto& m = r.metrics;
		md << "| " << r.engine
		   << " | " << (m.chartCount ? std::to_string(m.chartCount) : "—")
		   << " | " << Num(m.boundaryCutLength, 3)
		   << " | " << Num(m.meanPlanarityError, 4)
		   << " | " << Num(m.meanChartCompactness, 3)
		   << " | " << (m.fullCoverage ? "yes" : "no")
		   << " | " << (r.segmentation.completed ? Secs(r.segmentation.wallSeconds) : "—")
		   << " |\n";
	}

	// --- Parametrization ----------------------------------------------------
	md << "\n## Parametrization\n\n";
	md << "| engine | flips | sym-Dirichlet | L2 stretch | quasiconf | area dist | finite | time (s) |\n";
	md << "|---|--:|--:|--:|--:|--:|:--:|--:|\n";
	for (const auto& r : results) {
		const auto& m = r.metrics;
		md << "| " << r.engine
		   << " | " << IntOrDash(m.flipCount)
		   << " | " << Num(m.symDirichlet, 3)
		   << " | " << Num(m.stretchL2, 3)
		   << " | " << Num(m.quasiconformal, 3)
		   << " | " << Num(m.areaDistortionRatio, 3)
		   << " | " << (m.allFinite ? "yes" : "NO")
		   << " | " << (r.parametrization.completed ? Secs(r.parametrization.wallSeconds) : "—")
		   << " |\n";
	}

	// --- Packing ------------------------------------------------------------
	md << "\n## Packing\n\n";
	md << "| engine | occ (rect) | occ (tri) | pages | overlaps | time (s) |\n";
	md << "|---|--:|--:|--:|:--:|--:|\n";
	for (const auto& r : results) {
		const auto& m = r.metrics;
		md << "| " << r.engine
		   << " | " << Pct(m.occupancyRect)
		   << " | " << Pct(m.occupancyTri)
		   << " | " << (m.numPages ? std::to_string(m.numPages) : "—")
		   << " | " << (m.hasBboxOverlaps ? "yes" : "no")
		   << " | " << (r.packing.completed ? Secs(r.packing.wallSeconds) : "—")
		   << " |\n";
	}

	// --- Cross-cutting ------------------------------------------------------
	md << "\n## Cost & validity\n\n";
	md << "| engine | total time (s) | peak RSS (MB) | valid | note |\n";
	md << "|---|--:|--:|:--:|---|\n";
	for (const auto& r : results) {
		md << "| " << r.engine
		   << " | " << Secs(r.totalWallSeconds)
		   << " | " << Num(static_cast<double>(r.peakRssBytes) / (1024.0 * 1024.0), 1)
		   << " | " << (r.validOutput ? "yes" : "NO")
		   << " | " << r.note
		   << " |\n";
	}
	md << "\n";
	return md.str();
}

bool WriteReports(const BenchConfig& cfg,
                  const std::vector<EngineResult>& results)
{
	// Markdown
	{
		std::ofstream f(cfg.outDir + "/report.md");
		if (!f)
			return false;
		f << RenderMarkdown(cfg, results);
	}

	// JSON
	{
		std::ofstream f(cfg.outDir + "/report.json");
		if (!f)
			return false;
		f << "{\n";
		f << "  \"mesh\": " << J(cfg.meshPath) << ",\n";
		f << "  \"tier\": " << J(cfg.tier) << ",\n";
		f << "  \"resolution\": " << cfg.resolution << ",\n";
		f << "  \"stage\": " << J(cfg.stage) << ",\n";
		f << "  \"engines\": [\n";
		for (size_t i = 0; i < results.size(); ++i) {
			const auto& r = results[i];
			const auto& m = r.metrics;
			f << "    {\n";
			f << "      \"engine\": " << J(r.engine) << ",\n";
			f << "      \"valid\": " << (r.validOutput ? "true" : "false") << ",\n";
			f << "      \"peak_rss_bytes\": " << r.peakRssBytes << ",\n";
			f << "      \"time\": { \"segmentation\": " << JNum(r.segmentation.wallSeconds)
			  << ", \"parametrization\": " << JNum(r.parametrization.wallSeconds)
			  << ", \"packing\": " << JNum(r.packing.wallSeconds)
			  << ", \"total\": " << JNum(r.totalWallSeconds) << " },\n";
			f << "      \"segmentation\": { \"chart_count\": " << m.chartCount
			  << ", \"boundary_cut_length\": " << JNum(m.boundaryCutLength)
			  << ", \"mean_planarity_error\": " << JNum(m.meanPlanarityError)
			  << ", \"mean_chart_compactness\": " << JNum(m.meanChartCompactness)
			  << ", \"full_coverage\": " << (m.fullCoverage ? "true" : "false") << " },\n";
			f << "      \"parametrization\": { \"flip_count\": " << m.flipCount
			  << ", \"sym_dirichlet\": " << JNum(m.symDirichlet)
			  << ", \"stretch_l2\": " << JNum(m.stretchL2)
			  << ", \"quasiconformal\": " << JNum(m.quasiconformal)
			  << ", \"area_distortion_ratio\": " << JNum(m.areaDistortionRatio)
			  << ", \"all_finite\": " << (m.allFinite ? "true" : "false") << " },\n";
			f << "      \"packing\": { \"occupancy_rect\": " << JNum(m.occupancyRect)
			  << ", \"occupancy_tri\": " << JNum(m.occupancyTri)
			  << ", \"num_pages\": " << m.numPages
			  << ", \"has_bbox_overlaps\": " << (m.hasBboxOverlaps ? "true" : "false") << " },\n";
			f << "      \"note\": " << J(r.note) << "\n";
			f << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
		}
		f << "  ]\n";
		f << "}\n";
	}

	// CSV (one row per engine; flat columns)
	{
		std::ofstream f(cfg.outDir + "/report.csv");
		if (!f)
			return false;
		f << "engine,valid,chart_count,boundary_cut_length,mean_planarity_error,"
		     "mean_chart_compactness,full_coverage,flip_count,sym_dirichlet,stretch_l2,"
		     "quasiconformal,area_distortion_ratio,all_finite,occupancy_rect,occupancy_tri,"
		     "num_pages,has_bbox_overlaps,seg_time,param_time,pack_time,total_time,peak_rss_bytes\n";
		for (const auto& r : results) {
			const auto& m = r.metrics;
			auto c = [](double v) { return IsSet(v) ? std::to_string(v) : std::string(); };
			f << r.engine << ',' << (r.validOutput ? 1 : 0) << ',' << m.chartCount << ','
			  << c(m.boundaryCutLength) << ',' << c(m.meanPlanarityError) << ','
			  << c(m.meanChartCompactness) << ',' << (m.fullCoverage ? 1 : 0) << ','
			  << m.flipCount << ',' << c(m.symDirichlet) << ',' << c(m.stretchL2) << ','
			  << c(m.quasiconformal) << ',' << c(m.areaDistortionRatio) << ','
			  << (m.allFinite ? 1 : 0) << ',' << c(m.occupancyRect) << ','
			  << c(m.occupancyTri) << ',' << m.numPages << ','
			  << (m.hasBboxOverlaps ? 1 : 0) << ',' << r.segmentation.wallSeconds << ','
			  << r.parametrization.wallSeconds << ',' << r.packing.wallSeconds << ','
			  << r.totalWallSeconds << ',' << r.peakRssBytes << '\n';
		}
	}
	return true;
}

} // namespace hmbench
