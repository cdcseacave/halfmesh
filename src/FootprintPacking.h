/*
* FootprintPacking.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Internal footprint-packing entry for PackAtlas's fit-to-resolution search,
// which only asks whether the charts fit one page (see halfmesh/RectPacking.h).

#pragma once

#include <halfmesh/RectPacking.h>

#include <cstdint>
#include <vector>

namespace halfmesh::detail {

// Whether PackFootprints would place every mask on one page; each page width it
// tries is given up at the first footprint it leaves over. `footprintArea`
// returns the texels of all the block-quantized footprints, as unrotated.
bool FootprintsFitOnePage(const std::vector<cv::Mat>& masks, const FootprintPackParams& params, uint64_t& footprintArea);

} // namespace halfmesh::detail
