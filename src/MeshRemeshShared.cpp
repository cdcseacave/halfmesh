/*
* MeshRemeshShared.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "MeshRemeshShared.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <utility>
#include <vector>

namespace halfmesh {
namespace detail {

bool FairMesh(std::vector<Eigen::Matrix<double, 3, 1>>& points,
              const std::vector<Eigen::Matrix<int, 3, 1>>& tris,
              const std::vector<bool>& locked,
              unsigned k,
              bool cotangent)
{
	using SpMat = Eigen::SparseMatrix<double>;
	using Triplet = Eigen::Triplet<double>;

	const int n = static_cast<int>(points.size());
	if (n == 0 || k == 0 || static_cast<int>(locked.size()) != n)
		return false;

	std::vector<Triplet> ltrip;
	if (!cotangent) {
		// Undirected one-ring edges (deduped, deterministic order). Collect into a
		// contiguous vector (3 per triangle, ordered a<=b) then sort+unique: the
		// pair's lexicographic order matches std::map's iteration order exactly, so
		// the deduped edge list — hence the assembled Laplacian and the faired
		// positions — is byte-identical to the old std::map, with no per-edge node
		// allocation.
		std::vector<std::pair<int, int>> edges;
		edges.reserve(tris.size() * 3);
		for (const auto& f : tris) {
			for (int e = 0; e < 3; ++e) {
				int a = f[e];
				int b = f[(e + 1) % 3];
				if (a == b)
					continue;
				if (a > b)
					std::swap(a, b);
				edges.emplace_back(a, b);
			}
		}
		std::sort(edges.begin(), edges.end());
		edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

		// Uniform graph Laplacian L: L(i,i)=deg(i), L(i,j)=-1 per undirected edge.
		ltrip.reserve(edges.size() * 4);
		for (const auto& e : edges) {
			const int a = e.first;
			const int b = e.second;
			ltrip.emplace_back(a, b, -1.0);
			ltrip.emplace_back(b, a, -1.0);
			ltrip.emplace_back(a, a, 1.0);
			ltrip.emplace_back(b, b, 1.0);
		}
	} else {
		// Cotangent Laplacian: per undirected edge (i,j), w_ij = sum over the incident
		// triangles of cot(angle opposite the edge) / 2 (PMP/Liepa minimize_curvature).
		// Accumulate contributions into a sorted flat vector (deterministic), clamp
		// each edge weight to a small epsilon so obtuse-triangle negatives cannot
		// break SPD.
		std::vector<std::pair<std::pair<int, int>, double>> ew;
		ew.reserve(tris.size() * 3);
		for (const auto& f : tris) {
			for (int e = 0; e < 3; ++e) {
				const int i = f[e];
				const int j = f[(e + 1) % 3];
				const int c = f[(e + 2) % 3]; // apex opposite edge (i,j)
				if (i == j)
					continue;
				const Eigen::Matrix<double, 3, 1> u = points[i] - points[c];
				const Eigen::Matrix<double, 3, 1> v = points[j] - points[c];
				const double cross = u.cross(v).norm();
				const double cot = (cross > 0.0) ? (u.dot(v) / cross) : 0.0;
				int a = i, b = j;
				if (a > b)
					std::swap(a, b);
				ew.emplace_back(std::make_pair(a, b), 0.5 * cot);
			}
		}
		std::sort(ew.begin(), ew.end(),
		          [](const std::pair<std::pair<int, int>, double>& x,
		             const std::pair<std::pair<int, int>, double>& y) { return x.first < y.first; });
		constexpr double wMin = 1e-6;
		for (std::size_t i = 0; i < ew.size();) {
			std::size_t j = i;
			double wsum = 0.0;
			const std::pair<int, int> key = ew[i].first;
			while (j < ew.size() && ew[j].first == key) {
				wsum += ew[j].second;
				++j;
			}
			const double w = std::max(wsum, wMin);
			ltrip.emplace_back(key.first, key.second, -w);
			ltrip.emplace_back(key.second, key.first, -w);
			ltrip.emplace_back(key.first, key.first, w);
			ltrip.emplace_back(key.second, key.second, w);
			i = j;
		}
	}
	SpMat L(n, n);
	L.setFromTriplets(ltrip.begin(), ltrip.end()); // duplicate diagonal terms sum -> deg

	// A = L^k : k=1 membrane (||L x||), k=2 thin-plate (||L^2 x||), ...
	SpMat A = L;
	for (unsigned i = 1; i < k; ++i)
		A = (A * L).eval();

	// Partition vertices into free (solved) and locked (Dirichlet on the rhs).
	std::vector<int> idx(n, -1);
	std::vector<int> freev;
	freev.reserve(n);
	for (int v = 0; v < n; ++v) {
		if (!locked[v]) {
			idx[v] = static_cast<int>(freev.size());
			freev.push_back(v);
		}
	}
	const int nf = static_cast<int>(freev.size());
	if (nf == 0)
		return false;

	// rhs = -(A * X_locked) restricted to the free rows (X_locked has free rows zeroed).
	Eigen::MatrixXd xlock = Eigen::MatrixXd::Zero(n, 3);
	for (int v = 0; v < n; ++v)
		if (locked[v])
			xlock.row(v) = points[v].transpose();
	const Eigen::MatrixXd fullRhs = -(A * xlock);

	// Extract the free-free block A_ff.
	std::vector<Triplet> afftrip;
	afftrip.reserve(static_cast<size_t>(A.nonZeros()));
	for (int col = 0; col < A.outerSize(); ++col) {
		if (idx[col] < 0)
			continue; // locked column -> already folded into the rhs
		for (SpMat::InnerIterator it(A, col); it; ++it) {
			const int row = static_cast<int>(it.row());
			if (idx[row] < 0)
				continue; // locked row
			afftrip.emplace_back(idx[row], idx[col], it.value());
		}
	}
	SpMat aff(nf, nf);
	aff.setFromTriplets(afftrip.begin(), afftrip.end());

	Eigen::MatrixXd rhs(nf, 3);
	for (int i = 0; i < nf; ++i)
		rhs.row(i) = fullRhs.row(freev[i]);

	Eigen::SimplicialLDLT<SpMat> solver(aff);
	if (solver.info() != Eigen::Success)
		return false;
	const Eigen::MatrixXd x = solver.solve(rhs);
	if (solver.info() != Eigen::Success)
		return false;
	// LDLT reports Success for NaN-contaminated inputs (NaN locked positions
	// poison the rhs) and for numerically singular A^k — never commit a
	// non-finite solve; the doc'd contract is "false = positions left
	// unchanged" and callers fall back on it (hole-filler k=2 -> k=1 -> skip).
	if (!x.allFinite())
		return false;

	for (int i = 0; i < nf; ++i)
		points[freev[i]] = Eigen::Matrix<double, 3, 1>(x(i, 0), x(i, 1), x(i, 2));
	return true;
}

} // namespace detail
} // namespace halfmesh
