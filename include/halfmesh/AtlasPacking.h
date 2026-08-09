/*
* AtlasPacking.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// halfmesh/AtlasPacking.h — Module D of the atlas pipeline: chart packing.
//
// Packs the per-chart UV islands (already density-normalised by
// NormalizeChartDensity, halfmesh/AtlasCharting.h) into a single texture page and
// writes atlas-space [0,1] UVs back into mesh.faceTexcoords. The packer is a
// skyline bottom-left bin packer with a per-chart minimum-area-rectangle
// pre-orientation (rotating calipers) so diagonal/elongated charts pack tightly.
// AtlasParams and AtlasResult (shared) are declared in halfmesh/AtlasCharting.h.
#pragma once

#include <halfmesh/AtlasCharting.h> // AtlasParams, AtlasResult, Mesh
#include <halfmesh/Mesh.h>

#include <vector>

namespace halfmesh {

// Module D: pack density-normalized charts into atlas page(s).
// Assumes NormalizeChartDensity has already been called.
// Writes atlas-space [0,1] UVs into mesh.faceTexcoords.
AtlasResult PackAtlas(Mesh& mesh,
                      const std::vector<unsigned>& faceChart,
                      unsigned numCharts,
                      const AtlasParams& params = AtlasParams{});

} // namespace halfmesh
