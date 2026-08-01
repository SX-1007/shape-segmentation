#ifndef SEGMENTATION_TEST_H
#define SEGMENTATION_TEST_H

#include "ShapeDetect.h"
#include <opencv2/opencv.hpp>
#include <string>

// Pixel-level scores.  Shape-label accuracy alone cannot detect a removed rim
// or a mask containing only half of the physical sign.
struct MaskMetrics {
	double iou;
	double dice;
	double precision;
	double recall;
	double boundaryIoU;

	MaskMetrics();
};

MaskMetrics evaluateBinaryMask(const cv::Mat &prediction,
	const cv::Mat &groundTruth);

// Fast deterministic regression checks for metric edge cases, hole filling,
// the four ideal shape classes, and rebuilt-mask integrity.
int runShapeDetectionSelfTests();

// Build a deterministic, license-free stress set with exact pixel masks.  The
// signs are rendered over foliage, brick, sky, urban clutter, coloured
// distractors, shadows, low light and compression artefacts.
bool generateSyntheticBackgroundDataset(const std::string &root,
	int imageCount = 64);

// Evaluate the complete detector/refiner against images/ and masks/ below root.
// Predictions and a per-image CSV report are written below the same root.
int evaluateSegmentationDataset(const std::string &root,
	const ShapeParams &params, bool savePredictions = true);

// Evaluate against CamVid's hand-labelled SignSymbol pixels.  The downloaded
// archives are treated as read-only; all derived crops, masks, overlays and
// reports are written below outputRoot.  Each connected SignSymbol region is
// evaluated in a square natural-background context crop and normalised to
// 256x256, matching this module's sign-candidate input contract.
int evaluateCamVidDataset(const std::string &datasetRoot,
	const std::string &outputRoot, const ShapeParams &params,
	bool savePredictions = true, const std::string &split = "all",
	double contextScale = 3.0,
	const std::string &originalLabelRoot = std::string(),
	bool useBoxPrompt = false);

#endif
