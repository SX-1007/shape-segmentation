// ============================================================================
//  ShapeDetect.h
//  Module   : Shape detection of signs to support the segmentation
//  Component: Member 4 shape/segmentation-support module
//  Subject  : UCCC2513 Mini Project - Traffic sign detection
//
//  This module detects the geometric shape (circle / triangle / rectangle /
//  octagon) of every candidate region produced by the colour segmentation
//  stage, and uses that shape knowledge to
//     (1) reject false colour blobs (background objects that happen to be
//         red / blue / yellow),
//     (2) pick the real sign instead of blindly taking the "longest contour"
//         as done in the lecture examples (Ex8 / Ex4), and
//     (3) rebuild a clean, hole-free mask from the *idealised* shape so that
//         the final "mask AND original image" segmentation is complete.
//
//  How the colour members plug into it:
//         Mat  myMask = <red / blue / yellow segmentation of member 1..3>;
//         vector<ShapeInfo> cand = detectSignShapes(myMask, param);
//         int  k    = selectBestSign(cand, param);
//         Mat  mask = buildShapeMask(cand[k], image.size());
//         Mat  sign; image.copyTo(sign, mask);
// ============================================================================
#ifndef SHAPE_DETECT_H
#define SHAPE_DETECT_H

#include	<opencv2/opencv.hpp>
#include	<vector>
#include	<string>

using namespace cv;
using namespace std;

// ---------------------------------------------------------------------------
// The four sign shapes stated in the assignment specification.
// ---------------------------------------------------------------------------
enum SignShape {
	SHAPE_UNKNOWN = 0,
	SHAPE_TRIANGLE,
	SHAPE_RECTANGLE,
	SHAPE_OCTAGON,
	SHAPE_CIRCLE
};

// How a candidate was proposed.  Colour contours are fast and reliable when
// the paint is clean; edge contours and Hough circles recover signs whose
// colour is faded, fragmented, or printed on a larger carrier board.
enum ProposalSource {
	SOURCE_COLOR = 0,
	SOURCE_EDGE,
	SOURCE_HOUGH
};

// ---------------------------------------------------------------------------
// Everything measured / decided for one candidate region.
// ---------------------------------------------------------------------------
struct ShapeInfo {
	SignShape		shape;			// classification result
	double			score;			// confidence of the classification, 0..1
	int				colorId;		// 0 = red, 1 = blue, 2 = yellow, -1 = n/a

	vector<Point>	contour;		// raw boundary from findContours()
	vector<Point>	hull;			// convex hull of the boundary
	vector<Point>	poly;			// polygonal approximation (approxPolyDP)
	vector<Point>	ideal;			// idealised outline used to rebuild the mask

	Point2f			circleCenter;	// minEnclosingCircle() result
	float			circleRadius;
	RotatedRect		ellipse;		// AMS ellipse fit for perspective-viewed circles
	bool			ellipseValid;
	RotatedRect		box;			// minAreaRect() result
	Rect			bbox;			// upright bounding box

	double			area;			// contourArea() of the raw boundary
	double			perimeter;		// arcLength() of the raw boundary
	double			hullArea;		// area of the convex hull = area of the plate
	double			hullPerim;		// perimeter of the convex hull
	double			circularity;	// 4*pi*A / P^2 of the hull (1.0 for a circle)
	double			solidity;		// area / hullArea   (~1.0 for a filled plate)
	double			extent;			// area / bbox area
	double			aspect;			// long side / short side of minAreaRect (>= 1)

	// A sign whose colour is only a rim (the red ring of a speed limit) gives a
	// band-shaped boundary: it walks the outer edge and comes back along the
	// inner edge, so its perimeter is about twice the hull perimeter and its
	// solidity is low even though the plate itself is a perfect disc.
	double			rimRatio;		// perimeter / hullPerim  (~2.0 for a rim)
	bool			rimRescued;		// accepted through the rim rule

	// The three fit ratios below are measured after aspect normalisation, so
	// they are invariant to translation, rotation, scale and to the
	// foreshortening of a sign that is not seen head-on.
	double			cirFit;			// A / area of min enclosing circle
	double			rectFit;		// A / area of min area rectangle
	double			triFit;			// A / area of min enclosing triangle

	int				vertLo;			// corners at the fine tolerance   (1% of P)
	int				vertices;		// corners at the coarse tolerance (3% of P)
	int				vertVeryHi;		// corners after stronger simplification (5%)
	bool			touchesBorder;	// region runs out of the picture

	// Multi-cue evidence used by the advanced selector.  These measurements are
	// deliberately kept separate from the geometric classification confidence:
	// a mathematically good rectangle can still be a wall or a carrier board.
	ProposalSource	proposalSource;
	double			edgeSupport;	// fraction of the fitted outline supported by edges
	double			colorCoverage;	// coloured pixels / fitted plate area
	double			colorCapture;	// share of local colour evidence captured by plate
	double			contrast;		// Lab contrast across the fitted boundary, 0..1
	double			centerProximity;// 1 at crop centre, 0 near a corner
	double			selectionScore;	// fused ranking score, 0..1

	string			reject;			// why the region was thrown away ("" = kept)
};

// ---------------------------------------------------------------------------
// Tunable parameters of the module (all in one place so that they are easy to
// report, justify and re-tune).  The default values were obtained by a grid
// search over the 84 test images.
// ---------------------------------------------------------------------------
struct ShapeParams {
	// -- colour candidate stage --------------------------------------------
	double	thRed, thBlue, thYellow;	// colour-enhancement thresholds
	double	adaptRel;					// per-image adaptive share of the peak
	double	thFloor;					// lowest threshold the adaption may use
	double	yellowGreenTol;				// how much greener than red yellow may be
	double	darkSum;					// R+G+B below this is too dark to judge
	double	morphFrac;					// closing kernel, as a share of the image
	int		morphMin;					// smallest useful closing kernel, in pixels
	bool	fillHoles;					// close the pictogram holes inside a plate

	// -- second, coarser pass (welds a plate that survived only as fragments) -
	bool	multiScale;					// run the coarse pass at all
	double	weldFrac;					// welding kernel, as a share of the image
	int		weldMin;					// ... never smaller than this
	double	weldIoU;					// overlap above which the fine region wins
	bool	splitMerged;				// proposal-only opening to break thin bridges
	double	splitOpenFrac;			// opening kernel as a share of the image
	int		splitOpenMin;				// absolute minimum opening kernel

	// -- region filtering ---------------------------------------------------
	double	minAreaRatio;				// smallest region, as a share of the image
	double	maxAreaRatio;				// largest region
	double	minArea;					// absolute floor, in pixels
	double	minSolidity;				// traffic signs are convex plates
	double	maxAspect;					// reject poles, road markings, stripes

	// -- rim rule (a sign whose colour is only the ring around the plate) ----
	double	rimMinSolid;				// a rim still covers this much of the hull
	double	rimLo, rimHi;				// accepted perimeter / hullPerim window
	double	rimMinCirc;					// the hull of a rim sign stays compact

	// -- shape classification -----------------------------------------------
	double	epsLo, epsHi;				// approxPolyDP tolerances (share of P)
	double	epsFactor;					// tolerance used for the drawn polygon
	int		circleVerts;				// fine corners a disc is expected to keep
	double	wVert, wCirc;				// weight of the corner-count penalties
	double	octagonBias;				// small prior against the rare octagon
	double	rectGate, triGate, octGate;	// a model must meet its own descriptor
	double	minScore;					// smallest accepted confidence

	// -- sign selection -----------------------------------------------------
	double	innerMinArea;				// inner plate vs outer board, area share
	double	innerMaxArea;				// larger nested regions are usually graphics
	double	innerMinScore;				// ... and relative confidence
	double	selectionSizeWeight;		// scale evidence in already-cropped sign images
	double	centerPriorStrength;			// light prior for this crop-based module

	// -- advanced edge proposals and evidence fusion ------------------------
	bool	edgeProposals;				// add Canny/contour candidates
	bool	houghCircles;				// optional; off unless circle recall needs it
	double	edgeCloseFrac;			// close small gaps in the edge sketch
	int		edgeCloseMin;
	double	edgeToleranceFrac;		// outline-to-edge matching tolerance
	double	minEdgeSupport;			// reject unsupported edge-only models
	double	minColorCoverage;		// suppress uncoloured pictograms/digits
	double	dedupeIoU;				// merge duplicate colour/edge proposals

	// -- pixel-accurate, shape-constrained GrabCut refinement ---------------
	bool	refineWithGrabCut;
	int		grabCutIterations;
	double	grabCutOuterFrac;		// allowed motion outside the fitted outline
	double	grabCutInnerFrac;		// erosion used to make sure-foreground seeds
	double	refineMinAreaRatio;		// plausibility window relative to shape prior
	double	refineMaxAreaRatio;
	double	refineMinSpanRatio;		// reject a one-sided / half-object graph cut
	double	refineMinColorRecall;	// preserve reliable sign-colour paint, incl. rims

	ShapeParams();						// constructor fills in the defaults
};

// Why graph-cut refinement was accepted, repaired, or replaced by the shape
// prior.  These values make partial-mask failures observable during testing.
struct RefinementDiagnostics {
	bool	attempted;
	bool	accepted;
	bool	colorRepaired;
	int		priorArea;
	int		graphCutArea;
	int		finalArea;
	double	areaRatio;
	double	coreRecall;
	double	colorRecall;
	double	spanRatio;
	double	priorScale;			// >1 when a second, outer border was selected
	string	decision;

	RefinementDiagnostics();
};

// ---------------------------------------------------------------------------
// Public interface of the module
// ---------------------------------------------------------------------------

// Name / drawing colour of a shape (for legends and console reports).
const char*	shapeName(SignShape s);
Scalar		shapeColor(SignShape s);

// Stage 1 (support stage) - colour candidate generation.  Produces one cleaned
// binary mask per sign colour.  The red / blue / yellow members can replace
// this call with their own segmentation and feed the masks straight into
// detectSignShapes().
void	buildColorMasks(const Mat &bgr, const ShapeParams &p,
			Mat &redMask, Mat &blueMask, Mat &yellowMask);

// Convenience wrapper returning the union of the three masks (for display).
Mat		buildColorCandidateMask(const Mat &bgr, const ShapeParams &p,
			Mat *redMask = NULL, Mat *blueMask = NULL, Mat *yellowMask = NULL);

// Morphological clean-up of any binary mask before contour following.
Mat		cleanMask(const Mat &mask, const ShapeParams &p);

// Stage 2 - measure one contour and decide which shape it is.
// Returns false when the region is rejected (too small, not convex, ...).
bool	classifyContour(const vector<Point> &contour, Size imgSize,
			const ShapeParams &p, ShapeInfo &info);

// Stage 2 (batch) - boundary following + classification.
// Each colour is followed separately so that two touching regions of different
// colours (a red sign mounted on a blue board) are never merged into one blob.
// "rejected", when given, receives the regions that were thrown away together
// with the reason, so that a picture yielding no candidate can be diagnosed.
vector<ShapeInfo>	detectSignShapes(const Mat &mask, const ShapeParams &p,
						int colorId = -1, vector<ShapeInfo> *rejected = NULL);
vector<ShapeInfo>	detectSignShapes(const vector<Mat> &masks, const ShapeParams &p,
						vector<ShapeInfo> *rejected = NULL);

// Advanced route used by the demonstration: fuse the colour candidates above
// with candidates recovered from the image edges, measure boundary support and
// colour/contrast evidence, then suppress duplicate proposals.  The optional
// edge image is useful for a report/demo stage.
vector<ShapeInfo>	detectSignShapesAdvanced(const Mat &bgr,
						const vector<Mat> &masks, const ShapeParams &p,
						Mat *edgeImage = NULL,
						vector<ShapeInfo> *rejected = NULL);

// Build the adaptive Canny edge sketch used by the advanced proposal stage.
void	buildAdaptiveEdgeMap(const Mat &bgr, const ShapeParams &p, Mat &edges);

// Stage 3 - choose the region that is most likely to be the traffic sign.
// This replaces the "longest contour" assumption of the lecture examples.
// Returns -1 when no acceptable candidate exists.
int		selectBestSign(const vector<ShapeInfo> &list, const ShapeParams &p);

// Stage 4 - rebuild a solid mask from the idealised shape (a filled disc for a
// circle, a filled polygon otherwise).  This closes the ring-shaped colour mask
// of, e.g., a red speed-limit sign without needing floodFill() from a centre
// point that may fall outside the region.
Mat		buildShapeMask(const ShapeInfo &info, Size sz);

// Refine the ideal geometric prior with an automatically generated GrabCut
// trimap.  The physical sign boundary is allowed to contract/expand slightly,
// while pixels far outside the fitted shape are locked to background.  This is
// the final defence against the background wedges left by an enclosing model.
Mat		refineShapeMask(const Mat &bgr, const ShapeInfo &info,
			const vector<Mat> &colorMasks, const ShapeParams &p,
			Mat *trimapDebug = NULL,
			RefinementDiagnostics *diagnostics = NULL);

// Drawing / reporting helpers used by the demo program.
void	drawShape(Mat &canvas, const ShapeInfo &info, bool withLabel = true,
			int thickness = 2);
void	printShapeReport(const ShapeInfo &info, int index);
void	printRejectReport(const ShapeInfo &info, int index);

#endif
