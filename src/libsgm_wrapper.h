#ifndef __LIBSGM_WRAPPER_H__
#define __LIBSGM_WRAPPER_H__

#include "libsgm.h"
#include <memory>

namespace sgm {
	/**
	 * @brief LibSGMWrapper class which is wrapper for sgm::StereoSGM.
	 */
	class LibSGMWrapper {
	public:
		/**
		 * @param numDisparity Maximum disparity minus minimum disparity.
		 * @param P1 Penalty on the disparity change by plus or minus 1 between nieghbor pixels.
		 * @param P2 Penalty on the disparity change by more than 1 between neighbor pixels.
		 * @param uniquenessRatio Margin in ratio by which the best cost function value should be at least second one.
		 * @param subpixel Disparity value has 4 fractional bits if subpixel option is enabled.
		 * @param pathType Number of scanlines used in cost aggregation.
		 * @param minDisparity Minimum possible disparity value.
		 * @param lrMaxDiff Acceptable difference pixels which is used in LR check consistency. LR check consistency will be disabled if this value is set to negative.
		 */
		LibSGMWrapper(int numDisparity = 128, int P1 = 10, int P2 = 120, float uniquenessRatio = 0.95f,
				bool subpixel = false, PathType pathType = PathType::SCAN_8PATH, int minDisparity = 0, int lrMaxDiff = 1);
		~LibSGMWrapper();

		int getNumDisparities() const;
		int getP1() const;
		int getP2() const;
		float getUniquenessRatio() const;
		bool hasSubpixel() const;
		PathType getPathType() const;
		int getMinDisparity() const;
		int getLrMaxDiff() const;
		int getInvalidDisparity() const;

	private:
		struct Creator;
		std::unique_ptr<sgm::StereoSGM> sgm_;
		int numDisparity_;
		sgm::StereoSGM::Parameters param_;
		std::unique_ptr<Creator> prev_;
	};
}

#endif // __LIBSGM_WRAPPER_H__
