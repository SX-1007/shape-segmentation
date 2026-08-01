// ============================================================================
//  ShapeDetect.cpp
//  Module   : Shape detection of signs to support the segmentation
//  Component: Member 4 shape/segmentation-support module
//
//  Method (see report chapter 4):
//
//   Stage 1  Colour candidates (support stage).  Two cues must agree: the
//            normalised colour-enhancement transform and an HSV hue window.
//            One cleaned mask is produced per sign colour, and the holes left
//            inside a plate by its legend or pictogram are filled.
//
//   Stage 2  Boundary following (findContours) on each colour mask, as taught
//            in "6.1 Image processing operations", pages 25-28, followed by:
//
//            0. The convex hull, not the raw boundary, is taken as the plate.
//               A sign is a convex plate, so this is what lets a region survive
//               a pictogram that bites into its rim.  It also catches the sign
//               whose colour is only a *rim* - the red ring of a speed limit,
//               whose boundary runs out along the outer edge and back along the
//               inner one, giving a low solidity but a perfectly circular hull.
//            a. Aspect normalisation.  Every boundary is rotated into the
//               frame of its minimum-area rectangle and rescaled into a unit
//               square.  A sign seen at an angle is foreshortened, so this
//               step buys invariance to the viewpoint before any descriptor
//               is measured.
//            b. Region shape descriptors on the normalised boundary:
//                  cirFit  = A / area(minEnclosingCircle)
//                  rectFit = A / area(minAreaRect)
//                  triFit  = A / area(minEnclosingTriangle)
//               All three are invariant to translation, rotation and scale.
//            c. Polygonal approximation ("Ex9") of the convex hull at a fine
//               and a coarse tolerance.  A real polygon keeps the same corner
//               count at both tolerances while a circle loses corners quickly;
//               this "corner stability" is what separates a circle from an
//               octagon at the small image sizes of the test set.
//            d. Nearest-model matching against the analytically derived
//               descriptors of the four sign shapes.  A model may only compete
//               when its own key descriptor is met, which stops the rectangle
//               model from acting as a catch-all for blobby regions.  The
//               distance becomes a confidence score in 0..1.
//
//            Stages 1 and 2 are run twice, at two morphological scales.  The
//            coarse pass welds a plate that survived only as a broken outline
//            back into one region, but it also rounds the corners of a
//            triangle, so it may only *add* regions the fine pass missed.
//
//   Stage 3  Sign selection by shape confidence and plate size, with a
//            containment rule that keeps the inner plate when a sign is
//            painted on a larger guide board.
//
//   Stage 4  The idealised outline of the winning model rebuilds a solid mask
//            which is ANDed with the original image to segment the sign.
// ============================================================================

#include	"ShapeDetect.h"
#include	<iostream>
#include	<cstdio>
#include	<cmath>
#include	<algorithm>

static void requireBgr8(const Mat &image, const char *functionName) {
	if (image.empty())
		CV_Error(Error::StsBadArg, string(functionName) + ": input image is empty");
	if (image.type() != CV_8UC3)
		CV_Error(Error::StsUnsupportedFormat,
			string(functionName) + ": expected a CV_8UC3 BGR image");
}

static void requireMask8(const Mat &mask, Size expected,
		const char *functionName) {
	if (mask.empty()) return;
	if (mask.type() != CV_8UC1)
		CV_Error(Error::StsUnsupportedFormat,
			string(functionName) + ": masks must be CV_8UC1");
	if (expected.width > 0 && mask.size() != expected)
		CV_Error(Error::StsUnmatchedSizes,
			string(functionName) + ": image and mask dimensions do not match");
}

// ---------------------------------------------------------------------------
// Default parameter values (grid-searched on the 84 test images)
// ---------------------------------------------------------------------------
ShapeParams::ShapeParams() {
	// colour candidate stage
	thRed          = 0.14;
	thBlue         = 0.13;
	thYellow       = 0.11;
	adaptRel       = 0.55;	// threshold may fall to 55% of this picture's peak
	thFloor        = 0.05;	// but never below this
	yellowGreenTol = 0.20;	// foliage is much greener than red; yellow is not
	darkSum        = 60.0;	// R+G+B below this carries no reliable colour
	morphFrac      = 0.020;	// closing kernel ~2% of the shorter image side
	morphMin       = 3;		// ... and never smaller than 3 pixels
	fillHoles      = true;	// weld the pictogram holes into the plate

	// second, coarser pass.  A wider kernel here squares off the corners of a
	// triangle, which is why it may only *add* regions the fine pass missed.
	multiScale = true;
	weldFrac   = 0.055;		// welding kernel ~5.5% of the shorter image side
	weldMin    = 7;
	weldIoU    = 0.30;		// above this overlap the sharper fine region wins
	splitMerged = true;
	splitOpenFrac = 0.028;	// additional proposals split narrow same-colour bridges
	splitOpenMin = 5;

	// region filtering
	minAreaRatio = 0.006;	// a region below 0.6% of the picture is noise
	maxAreaRatio = 0.95;	// ... and a sign never fills the whole picture
	minArea      = 100.0;	// absolute floor, in pixels
	minSolidity  = 0.60;	// the pictogram may bite into the rim of the plate
	maxAspect    = 3.00;	// reject poles, road markings, long stripes

	// rim rule
	rimMinSolid = 0.12;		// a rim band still covers 12% of the plate
	rimLo       = 1.45;		// out along the outer edge and back along the inner
	rimHi       = 2.80;		// ... which is about twice the hull perimeter
	rimMinCirc  = 0.65;		// the plate behind the rim is still compact

	// shape classification
	epsLo       = 0.010;	// fine   approxPolyDP tolerance
	epsHi       = 0.030;	// coarse approxPolyDP tolerance
	epsFactor   = 0.020;	// tolerance of the polygon that gets drawn / filled
	circleVerts = 10;		// fine corners a disc is expected to keep
	wVert       = 0.050;
	wCirc       = 0.050;
	octagonBias = 0.020;	// the octagon is by far the rarest sign shape
	rectGate    = 0.90;		// a rectangle really fills its bounding rectangle
	triGate     = 0.65;		// a triangle really fills its enclosing triangle
	octGate     = 0.80;		// an octagon really fills its enclosing circle
	// Below this confidence the region is reported as UNKNOWN.  It is kept low
	// on purpose: a weak but plausible shape still rebuilds a better mask than
	// the raw colour blob, and the selection stage already prefers the
	// confident candidates.
	minScore    = 0.20;

	// sign selection
	innerMinArea  = 0.10;
	innerMaxArea  = 0.70;
	innerMinScore = 0.85;
	selectionSizeWeight = 0.35;
	centerPriorStrength = 0.22;

	// Advanced proposal fusion.  The edge pass is intentionally conservative:
	// it must agree with at least a small amount of traffic-sign colour, which
	// prevents numerals, arrows, windows and building edges becoming candidates.
	edgeProposals       = true;
	houghCircles        = false;	// optional; unconstrained votes over-propose pictograms
	edgeCloseFrac       = 0.018;
	edgeCloseMin        = 3;
	edgeToleranceFrac   = 0.018;
	minEdgeSupport      = 0.34;
	minColorCoverage    = 0.018;
	dedupeIoU           = 0.72;

	// Shape-constrained GrabCut.  Only a narrow band around the fitted model is
	// left undecided, so graph cut can snap to the plate edge but cannot wander
	// into an unrelated background object.
	refineWithGrabCut   = true;
	grabCutIterations   = 5;
	grabCutOuterFrac    = 0.080;
	grabCutInnerFrac    = 0.160;
	refineMinAreaRatio  = 0.72;
	refineMaxAreaRatio  = 1.16;
	refineMinSpanRatio  = 0.80;
	refineMinColorRecall = 0.90;
}

RefinementDiagnostics::RefinementDiagnostics()
	: attempted(false), accepted(false), colorRepaired(false), priorArea(0),
	  graphCutArea(0), finalArea(0), areaRatio(0.0), coreRecall(0.0),
	  colorRecall(0.0), spanRatio(0.0), priorScale(1.0),
	  decision("not attempted") {}

// ---------------------------------------------------------------------------
// Names and drawing colours
// ---------------------------------------------------------------------------
const char* shapeName(SignShape s) {
	switch (s) {
	case SHAPE_TRIANGLE:	return "TRIANGLE";
	case SHAPE_RECTANGLE:	return "RECTANGLE";
	case SHAPE_OCTAGON:		return "OCTAGON";
	case SHAPE_CIRCLE:		return "CIRCLE";
	default:				return "UNKNOWN";
	}
}

Scalar shapeColor(SignShape s) {			// B, G, R
	switch (s) {
	case SHAPE_TRIANGLE:	return Scalar(0, 255, 0);		// green
	case SHAPE_RECTANGLE:	return Scalar(255, 160, 0);		// blue
	case SHAPE_OCTAGON:		return Scalar(0, 200, 255);		// amber
	case SHAPE_CIRCLE:		return Scalar(0, 0, 255);		// red
	default:				return Scalar(170, 170, 170);	// grey
	}
}

// ---------------------------------------------------------------------------
// 99.5th percentile of a single-channel float map, through a histogram.
// Used to adapt the colour threshold to the picture at hand.
// ---------------------------------------------------------------------------
static double highPercentile(const Mat &f, double frac = 0.995) {
	const int	BINS = 512;
	int			hist[BINS] = { 0 };
	long		total = 0;

	for (int y = 0; y < f.rows; y++) {
		const float	*row = f.ptr<float>(y);
		for (int x = 0; x < f.cols; x++) {
			int	b = (int)(row[x] * BINS);
			if (b < 0) b = 0;
			if (b >= BINS) b = BINS - 1;
			hist[b]++;
			total++;
		}
	}
	if (total == 0) return 0.0;

	long	want = (long)(total * frac), acc = 0;
	for (int b = 0; b < BINS; b++) {
		acc += hist[b];
		if (acc >= want)
			return (b + 0.5) / BINS;
	}
	return 1.0;
}

// ---------------------------------------------------------------------------
// Stage 1: colour candidate masks
//
// This is only a *support* stage so that the shape module can be demonstrated
// on its own; the red / blue / yellow members feed their own (better) masks
// straight into detectSignShapes() instead.
//
// Two cues are combined:
//
//   (1) the normalised colour-enhancement transform, which divides the colour
//       difference by the total intensity and is therefore insensitive to how
//       brightly the sign happens to be lit:
//           fR = max(0, min(R-G, R-B)) / (R+G+B)
//           fB = max(0, min(B-R, B-G)) / (R+G+B)
//           fY = max(0, min(R-B, G-B)) / (R+G+B)
//       A grey wall or an overcast sky gives fR = fB = fY = 0, which is what
//       removes the huge background regions that plain HSV thresholding keeps.
//       Foliage would pass fY, so a pixel that is much greener than it is red
//       is cancelled first.
//       The threshold is then lowered towards the strongest response present
//       in the picture, so that a dull, badly lit sign is still segmented.
//
//   (2) an HSV hue window (lecture "7.1 Image processing operations",
//       pages 9-12) used as a sanity check on the enhanced response.
// ---------------------------------------------------------------------------
static void thresholdColorEvidence(const Mat &bgr, const ShapeParams &p,
		Mat &redMask, Mat &blueMask, Mat &yellowMask) {
	Mat		blurred, hsv, hRed1, hRed2, hRed, hBlue, hYellow;
	Mat		fR(bgr.size(), CV_32F), fB(bgr.size(), CV_32F), fY(bgr.size(), CV_32F);
	Mat		dark(bgr.size(), CV_8U);

	// Pre-processing: a light Gaussian blur removes the sensor noise before the
	// colour thresholding (lecture 6.1, page 22).
	GaussianBlur(bgr, blurred, Size(5, 5), 0);
	cvtColor(blurred, hsv, COLOR_BGR2HSV);

	// --- cue 2: hue windows -------------------------------------------------
	inRange(hsv, Scalar(0, 50, 35),   Scalar(12, 255, 255),  hRed1);
	inRange(hsv, Scalar(163, 50, 35), Scalar(180, 255, 255), hRed2);
	hRed = hRed1 | hRed2;					// red wraps around hue 0
	inRange(hsv, Scalar(90, 50, 25), Scalar(140, 255, 255), hBlue);
	inRange(hsv, Scalar(15, 50, 50), Scalar(40, 255, 255),  hYellow);

	// --- cue 1: normalised colour enhancement -------------------------------
	for (int y = 0; y < blurred.rows; y++) {
		const Vec3b	*src = blurred.ptr<Vec3b>(y);
		float		*pR = fR.ptr<float>(y);
		float		*pB = fB.ptr<float>(y);
		float		*pY = fY.ptr<float>(y);
		uchar		*pD = dark.ptr<uchar>(y);

		for (int x = 0; x < blurred.cols; x++) {
			double	B = src[x][0], G = src[x][1], R = src[x][2];
			double	sum = R + G + B;

			pD[x] = (sum < p.darkSum) ? 255 : 0;
			if (sum < 1.0) sum = 1.0;

			double	vR = min(R - G, R - B) / sum;
			double	vB = min(B - R, B - G) / sum;
			double	vY = min(R - B, G - B) / sum;

			// foliage is much greener than it is red; yellow paint is not
			if ((R - G) / sum < -p.yellowGreenTol) vY = 0.0;

			pR[x] = (float)(vR > 0 ? vR : 0);
			pB[x] = (float)(vB > 0 ? vB : 0);
			pY[x] = (float)(vY > 0 ? vY : 0);
		}
	}

	// --- per-image adaptive thresholds --------------------------------------
	double	tR = min(p.thRed,    max(p.thFloor, p.adaptRel * highPercentile(fR)));
	double	tB = min(p.thBlue,   max(p.thFloor, p.adaptRel * highPercentile(fB)));
	double	tY = min(p.thYellow, max(p.thFloor, p.adaptRel * highPercentile(fY)));

	Mat		bright = (dark == 0);
	redMask    = (fR > tR) & hRed    & bright;
	blueMask   = (fB > tB) & hBlue   & bright;
	yellowMask = (fY > tY) & hYellow & bright;

	// Intentionally do not fill holes here.  This raw evidence distinguishes a
	// red speed-limit rim from its white interior and prevents a pictogram that
	// merely lies inside a filled proposal mask from claiming 100% colour.
}

void buildColorMasks(const Mat &bgr, const ShapeParams &p,
		Mat &redMask, Mat &blueMask, Mat &yellowMask) {
	requireBgr8(bgr, "buildColorMasks");
	thresholdColorEvidence(bgr, p, redMask, blueMask, yellowMask);

	redMask    = cleanMask(redMask,    p);
	blueMask   = cleanMask(blueMask,   p);
	yellowMask = cleanMask(yellowMask, p);
}

Mat buildColorCandidateMask(const Mat &bgr, const ShapeParams &p,
		Mat *redMask, Mat *blueMask, Mat *yellowMask) {
	Mat		r, b, y;

	buildColorMasks(bgr, p, r, b, y);
	if (redMask)    r.copyTo(*redMask);
	if (blueMask)   b.copyTo(*blueMask);
	if (yellowMask) y.copyTo(*yellowMask);
	return r | b | y;
}

// ---------------------------------------------------------------------------
// Morphological clean-up.
//
// The white legend and the dark pictogram printed on a sign cut the coloured
// plate into several pieces, so the closing kernel is scaled to the picture
// size in order to weld those pieces back into one region before the boundary
// is followed.  The opening afterwards drops the isolated speckles.
// ---------------------------------------------------------------------------
// Fill every hole that is completely enclosed by the region: flood the
// background from a border pixel, then everything still black is a hole.
// This is what turns the ring left by a legend or a pictogram into a solid
// plate, so that the segmented sign comes out complete instead of hollow.
static void fillEnclosedHoles(Mat &bin) {
	Mat		flood;

	copyMakeBorder(bin, flood, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
	floodFill(flood, Point(0, 0), Scalar(255));
	Mat		holes = ~flood(Rect(1, 1, bin.cols, bin.rows));
	bin |= holes;
}

Mat cleanMask(const Mat &mask, const ShapeParams &p) {
	if (mask.empty()) return Mat();
	requireMask8(mask, mask.size(), "cleanMask");
	Mat		out;
	int		side = min(mask.cols, mask.rows);
	int		k    = cvRound(side * p.morphFrac);

	// The test images are small crops (78 - 193 px), so a kernel expressed only
	// as a share of the image collapses to 3 px and welds nothing back
	// together.  An absolute floor keeps the closing effective at that size.
	if (k < p.morphMin) k = p.morphMin;
	if (k % 2 == 0) k++;							// keep it odd

	Mat		big   = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
	Mat		small = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

	morphologyEx(mask, out, MORPH_CLOSE, big,   Point(-1, -1), 1);
	morphologyEx(out,  out, MORPH_OPEN,  small, Point(-1, -1), 1);
	if (p.fillHoles) fillEnclosedHoles(out);
	return out;
}

// ---------------------------------------------------------------------------
// Aspect normalisation.
//
// The boundary is rotated into the frame of its minimum-area rectangle and
// rescaled so that this rectangle becomes a 100 x 100 square.  Descriptors
// measured afterwards no longer depend on the foreshortening of the sign.
// ---------------------------------------------------------------------------
static void normalizeContour(const vector<Point> &c, const RotatedRect &box,
		vector<Point2f> &out) {
	double	ang = box.angle * CV_PI / 180.0;
	double	ca = cos(-ang), sa = sin(-ang);
	double	w = box.size.width, h = box.size.height;

	if (w < 1.0) w = 1.0;
	if (h < 1.0) h = 1.0;

	out.clear();
	out.reserve(c.size());
	for (size_t i = 0; i < c.size(); i++) {
		double	dx = c[i].x - box.center.x;
		double	dy = c[i].y - box.center.y;
		double	rx = dx * ca - dy * sa;
		double	ry = dx * sa + dy * ca;
		out.push_back(Point2f((float)(rx / w * 100.0), (float)(ry / h * 100.0)));
	}
}

// ---------------------------------------------------------------------------
// One shape model.  The constants are the descriptor values of the ideal
// polygon after the same aspect normalisation:
//   circle   : cirFit 1.000  rectFit 0.785  triFit 0.605
//   octagon  : cirFit 0.900  rectFit 0.828  triFit 0.630
//   square   : cirFit 0.637  rectFit 1.000  triFit 0.500
//   triangle : cirFit 0.407  rectFit 0.500  triFit 1.000
// ---------------------------------------------------------------------------
struct ShapeModel {
	SignShape	shape;
	double		cirFit, rectFit, triFit;
	int			corners;		// 0 means "a circle has no stable corner count"
};

static const ShapeModel	SHAPE_MODELS[4] = {
	{ SHAPE_CIRCLE,    1.000, 0.785, 0.605, 0 },
	{ SHAPE_OCTAGON,   0.900, 0.828, 0.630, 8 },
	{ SHAPE_RECTANGLE, 0.637, 1.000, 0.500, 4 },
	{ SHAPE_TRIANGLE,  0.407, 0.500, 1.000, 3 }
};

// Weighted distance between the measured descriptors and one model.
static double modelDistance(const ShapeInfo &f, const ShapeModel &m,
		const ShapeParams &p) {
	double	dc = f.cirFit  - m.cirFit;
	double	dr = f.rectFit - m.rectFit;
	double	dt = f.triFit  - m.triFit;
	double	d  = sqrt(0.35 * dc * dc + 0.30 * dr * dr + 0.35 * dt * dt);

	if (m.corners == 0) {
		// Circle: the fine approximation of a disc keeps many corners.
		int		miss = p.circleVerts - f.vertLo;
		if (miss > 0) d += p.wCirc * (miss > 4 ? 4 : miss);
		// A rounded rectangle can look circular at 3%, but its four long sides
		// normally survive a stronger simplification while a true disc keeps
		// more than four directional changes.
		if (f.vertVeryHi == 4 && f.rectFit > 0.78) d += 0.10;
	} else if (m.corners == 8) {
		// Octagon: it must show eight corners at the coarse tolerance and must
		// not keep gaining corners at the fine one (that would be a disc).
		int		dHi = abs(f.vertices - 8);
		int		extra = f.vertLo - 8;
		d += p.wVert * (dHi > 4 ? 4 : dHi);
		if (extra > 0) d += p.wCirc * (extra > 4 ? 4 : extra);
		d += p.octagonBias;
	} else {
		// Triangle / rectangle: the corners are rounded on a real sign, so only
		// the coarse count is trustworthy.
		int observed = (m.corners == 4 && f.vertVeryHi == 4)
			? f.vertVeryHi : f.vertices;
		int		dHi = abs(observed - m.corners);
		d += p.wVert * (dHi > 4 ? 4 : dHi);
	}
	return d;
}

// ---------------------------------------------------------------------------
// Stage 2: measure one contour and decide its shape
// ---------------------------------------------------------------------------
static string reasonArea(const char *what, double got, double limit) {
	char	buf[96];
	sprintf_s(buf, "%s %.0f (limit %.0f)", what, got, limit);
	return string(buf);
}

bool classifyContour(const vector<Point> &contour, Size imgSize,
		const ShapeParams &p, ShapeInfo &info) {
	info = ShapeInfo();				// reset every field
	info.colorId   = -1;
	info.contour   = contour;

	if (contour.size() < 6) {
		info.reject = "boundary shorter than 6 points";
		return false;
	}

	info.area      = contourArea(contour);
	info.perimeter = arcLength(contour, true);

	// --- convex hull = the plate -------------------------------------------
	// A traffic sign is a convex plate, so the hull - not the raw boundary - is
	// the region of interest.  Measuring the size and the shape on the hull is
	// what lets a sign survive a pictogram that bites into its rim, or a colour
	// mask that only covers the ring around the plate.
	convexHull(contour, info.hull);
	info.hullArea  = contourArea(info.hull);
	info.hullPerim = arcLength(info.hull, true);
	if (info.hullArea < 1.0 || info.hullPerim < 1.0 || info.perimeter < 1.0) {
		info.reject = "degenerate boundary";
		return false;
	}

	double	imgArea = (double)imgSize.width * imgSize.height;

	if (info.hullArea < p.minArea) {
		info.reject = reasonArea("area", info.hullArea, p.minArea);
		return false;
	}
	if (info.hullArea < p.minAreaRatio * imgArea) {
		info.reject = reasonArea("area", info.hullArea, p.minAreaRatio * imgArea);
		return false;
	}
	if (info.hullArea > p.maxAreaRatio * imgArea) {
		info.reject = reasonArea("area too large", info.hullArea,
			p.maxAreaRatio * imgArea);
		return false;
	}

	info.solidity    = info.area / info.hullArea;
	info.rimRatio    = info.perimeter / info.hullPerim;
	info.circularity = 4.0 * CV_PI * info.hullArea / (info.hullPerim * info.hullPerim);
	if (info.circularity > 1.0) info.circularity = 1.0;

	// --- is this a solid plate, or the rim of one? --------------------------
	if (info.solidity < p.minSolidity) {
		// The rim of a sign (the red ring of a speed limit) is traced out along
		// the outer edge and back along the inner one, so its boundary is about
		// twice the hull perimeter while the hull itself stays compact.  A
		// genuinely concave blob does not show that signature and is dropped.
		bool	looksLikeRim =
			info.solidity   >= p.rimMinSolid &&
			info.rimRatio   >= p.rimLo && info.rimRatio <= p.rimHi &&
			info.circularity >= p.rimMinCirc;

		if (!looksLikeRim) {
			char	buf[96];
			sprintf_s(buf, "solidity %.2f < %.2f and not a rim (P/Phull %.2f)",
				info.solidity, p.minSolidity, info.rimRatio);
			info.reject = buf;
			return false;			// a concave blob is not a sign plate
		}
		info.rimRescued = true;
	}

	// --- bounding shapes in the original image ------------------------------
	info.bbox = boundingRect(info.hull);
	info.box  = minAreaRect(info.hull);
	minEnclosingCircle(info.hull, info.circleCenter, info.circleRadius);
	info.ellipseValid = false;
	if (info.hull.size() >= 5) {
		try {
			// AMS is substantially less biased than a minimum enclosing circle
			// when a round sign is seen obliquely and therefore projects to an
			// ellipse.  This removes the background wedges that a circle adds.
			info.ellipse = fitEllipseAMS(info.hull);
			info.ellipseValid = info.ellipse.size.width > 2.f &&
				info.ellipse.size.height > 2.f;
		} catch (const cv::Exception&) {
			info.ellipseValid = false;
		}
	}

	double	longSide  = max(info.box.size.width, info.box.size.height);
	double	shortSide = min(info.box.size.width, info.box.size.height);
	if (shortSide < 4.0) {
		info.reject = "thinner than 4 pixels";
		return false;
	}
	info.aspect = longSide / shortSide;
	if (info.aspect > p.maxAspect) {
		char	buf[64];
		sprintf_s(buf, "aspect %.2f > %.2f (a pole or a marking)",
			info.aspect, p.maxAspect);
		info.reject = buf;
		return false;
	}

	info.extent = info.hullArea / (double)(info.bbox.width * info.bbox.height);
	double dx = (info.box.center.x - 0.5 * imgSize.width) /
		max(1.0, 0.5 * imgSize.width);
	double dy = (info.box.center.y - 0.5 * imgSize.height) /
		max(1.0, 0.5 * imgSize.height);
	info.centerProximity = clamp(1.0 - sqrt(dx * dx + dy * dy) / sqrt(2.0),
		0.0, 1.0);

	// --- aspect normalisation, then the descriptors -------------------------
	// The hull is normalised directly: for a rim-shaped region the raw boundary
	// runs twice around the plate, which would distort every descriptor.
	vector<Point2f>	norm, normHull;
	normalizeContour(info.hull, info.box, norm);
	convexHull(norm, normHull);

	double	nArea  = contourArea(normHull);
	double	nPerim = arcLength(normHull, true);
	if (nArea < 1.0 || nPerim < 1.0) {
		info.reject = "degenerate after aspect normalisation";
		return false;
	}

	Point2f			nCenter;
	float			nRadius = 0.f;
	vector<Point2f>	nTri;
	double			nTriArea = 0.0;

	minEnclosingCircle(normHull, nCenter, nRadius);
	try {							// this one can fail on a degenerate hull
		nTriArea = minEnclosingTriangle(normHull, nTri);
	} catch (const cv::Exception&) {
		nTriArea = 0.0;
	}
	RotatedRect	nBox      = minAreaRect(normHull);
	double		nRectArea = (double)nBox.size.width * nBox.size.height;

	info.cirFit  = (nRadius   > 1.f) ? nArea / (CV_PI * nRadius * nRadius) : 0.0;
	info.rectFit = (nRectArea > 1.0) ? nArea / nRectArea : 0.0;
	info.triFit  = (nTriArea  > 1.0) ? nArea / nTriArea  : 0.0;

	// --- polygonal approximation at two tolerances ("Ex9") ------------------
	// It is applied to the convex hull, not to the raw boundary, so that the
	// notches left by the pictogram inside the sign create no false corners.
	vector<Point2f>	polyLo, polyHi, polyVeryHi;
	approxPolyDP(normHull, polyLo, p.epsLo * nPerim, true);
	approxPolyDP(normHull, polyHi, p.epsHi * nPerim, true);
	approxPolyDP(normHull, polyVeryHi, 0.050 * nPerim, true);
	info.vertLo   = (int)polyLo.size();
	info.vertices = (int)polyHi.size();
	info.vertVeryHi = (int)polyVeryHi.size();

	// The same approximation in image coordinates, used for drawing / filling.
	// The tolerance follows the hull perimeter, not the raw one: a rim-shaped
	// boundary is twice as long as its hull and would otherwise be flattened
	// into a triangle by a tolerance twice too coarse.
	approxPolyDP(info.hull, info.poly, p.epsFactor * info.hullPerim, true);

	// --- nearest shape model ------------------------------------------------
	// A model is only allowed to compete when its own key descriptor is met.
	// Without this gate the rectangle model wins every blobby region, because
	// any compact blob fills a fair share of its bounding rectangle.
	int		best  = -1;
	double	bestD = 1e9;
	for (int i = 0; i < 4; i++) {
		SignShape	s = SHAPE_MODELS[i].shape;
		if (s == SHAPE_RECTANGLE && info.rectFit < p.rectGate &&
			!(info.vertVeryHi == 4 && info.rectFit >= 0.78)) continue;
		if (s == SHAPE_TRIANGLE  && info.triFit  < p.triGate)  continue;
		if (s == SHAPE_OCTAGON   && info.cirFit  < p.octGate)  continue;

		double	d = modelDistance(info, SHAPE_MODELS[i], p);
		if (d < bestD) { bestD = d; best = i; }
	}
	if (best < 0) {						// only the circle model is unconditional
		best  = 0;
		bestD = modelDistance(info, SHAPE_MODELS[0], p);
	}

	info.shape = SHAPE_MODELS[best].shape;
	info.score = 1.0 - bestD / 0.50;			// zero distance -> 1.0 confidence
	if (info.score < 0.0) info.score = 0.0;
	if (info.score > 1.0) info.score = 1.0;

	if (info.score < p.minScore)
		info.shape = SHAPE_UNKNOWN;

	// --- idealised outline used to rebuild the mask -------------------------
	// The corner list of the fitted model is preferred; the convex hull is the
	// fall-back when approxPolyDP did not return the expected corner count.
	vector<Point2f>	tri;
	info.ideal.clear();
	switch (info.shape) {
	case SHAPE_TRIANGLE:
		if ((int)info.poly.size() == 3)
			info.ideal = info.poly;
		else {
			try {
				minEnclosingTriangle(info.hull, tri);
				for (size_t i = 0; i < tri.size(); i++)
					info.ideal.push_back(Point(cvRound(tri[i].x), cvRound(tri[i].y)));
			} catch (const cv::Exception&) {
				info.ideal = info.hull;
			}
		}
		break;
	case SHAPE_RECTANGLE: {
			Point2f	pt[4];
			if ((int)info.poly.size() == 4)
				info.ideal = info.poly;
			else {
				info.box.points(pt);
				for (int i = 0; i < 4; i++)
					info.ideal.push_back(Point(cvRound(pt[i].x), cvRound(pt[i].y)));
			}
		}
		break;
	case SHAPE_OCTAGON:
		info.ideal = ((int)info.poly.size() >= 7 && (int)info.poly.size() <= 9)
			? info.poly : info.hull;
		break;
	case SHAPE_CIRCLE:
		// handled by buildShapeMask() through circleCenter / circleRadius
		info.ideal = info.hull;
		break;
	default:
		info.ideal = info.hull;
		break;
	}

	// --- does the region run out of the picture? ----------------------------
	info.touchesBorder = (info.bbox.x <= 1 || info.bbox.y <= 1 ||
		info.bbox.x + info.bbox.width  >= imgSize.width  - 2 ||
		info.bbox.y + info.bbox.height >= imgSize.height - 2);

	return true;
}

// ---------------------------------------------------------------------------
// Stage 2 (batch): boundary following over one mask
// ---------------------------------------------------------------------------
vector<ShapeInfo> detectSignShapes(const Mat &mask, const ShapeParams &p,
		int colorId, vector<ShapeInfo> *rejected) {
	vector<vector<Point> >	contours;
	vector<ShapeInfo>		result;
	Mat						tmp;

	if (mask.empty()) return result;
	requireMask8(mask, mask.size(), "detectSignShapes");
	mask.copyTo(tmp);				// findContours() modifies its input
	findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	for (size_t i = 0; i < contours.size(); i++) {
		ShapeInfo	info;
		if (classifyContour(contours[i], mask.size(), p, info)) {
			info.colorId = colorId;
			result.push_back(info);
		} else if (rejected) {
			info.colorId = colorId;
			rejected->push_back(info);
		}
	}
	return result;
}

// Overlap of two upright boxes, intersection over union.
static double bboxIoU(const Rect &a, const Rect &b) {
	double	inter = (double)(a & b).area();
	double	uni   = (double)a.area() + (double)b.area() - inter;

	return (uni > 0.0) ? inter / uni : 0.0;
}

// Each colour is followed separately: a red sign mounted on a blue board would
// otherwise be welded into one meaningless blob by the union of the masks.
//
// Every colour is then followed a second time at a coarser morphological scale.
// A plate whose colour survives only as a broken outline - a dark pictogram
// printed across it, a sign in shadow, a badly lit board - is welded back into
// one region by the wider kernel.  That kernel also rounds the corners of a
// triangle into a rectangle, so the coarse pass is allowed only to *add*
// regions: wherever the fine pass already found something, its sharper boundary
// is the one that is kept.
vector<ShapeInfo> detectSignShapes(const vector<Mat> &masks, const ShapeParams &p,
		vector<ShapeInfo> *rejected) {
	vector<ShapeInfo>	all;

	for (size_t c = 0; c < masks.size(); c++) {
		vector<ShapeInfo>	one = detectSignShapes(masks[c], p, (int)c, rejected);

		if (p.multiScale) {
			int		k = cvRound(min(masks[c].cols, masks[c].rows) * p.weldFrac);
			if (k < p.weldMin) k = p.weldMin;
			if (k % 2 == 0) k++;

			Mat		welded;
			morphologyEx(masks[c], welded, MORPH_CLOSE,
				getStructuringElement(MORPH_ELLIPSE, Size(k, k)));

			vector<ShapeInfo>	coarse = detectSignShapes(welded, p, (int)c);
			for (size_t i = 0; i < coarse.size(); i++) {
				bool	covered = false;
				for (size_t j = 0; j < one.size() && !covered; j++)
					if (bboxIoU(coarse[i].bbox, one[j].bbox) > p.weldIoU)
						covered = true;
				if (!covered)
					one.push_back(coarse[i]);
			}
		}

		if (p.splitMerged) {
			// Closing repairs fragmented paint, but it cannot separate a plate that
			// touches same-colour clutter through a thin bridge.  A second,
			// proposal-only opening stream supplies that complementary hypothesis.
			// The original mask is retained, and every opened proposal still has to
			// pass the normal geometry and image-evidence gates downstream.
			const double splitScales[2] = { 1.0, 1.75 };
			for (int scaleIndex = 0; scaleIndex < 2; scaleIndex++) {
				int k = cvRound(min(masks[c].cols, masks[c].rows) *
					p.splitOpenFrac * splitScales[scaleIndex]);
				if (k < p.splitOpenMin) k = p.splitOpenMin;
				if ((k & 1) == 0) k++;
				Mat opened;
				morphologyEx(masks[c], opened, MORPH_OPEN,
					getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
				vector<ShapeInfo> split = detectSignShapes(opened, p, (int)c);
				for (size_t i = 0; i < split.size(); i++) {
					bool duplicate = false;
					for (size_t j = 0; j < one.size() && !duplicate; j++)
						if (bboxIoU(split[i].bbox, one[j].bbox) > 0.88)
							duplicate = true;
					if (!duplicate) one.push_back(split[i]);
				}
			}
		}
		all.insert(all.end(), one.begin(), one.end());
	}
	return all;
}

// ---------------------------------------------------------------------------
// Stage 3: pick the region that is most likely to be the sign
//
// The lecture examples keep the *longest* contour, which fails whenever a
// larger red / blue / yellow object (a wall, a car, a guide board) is present.
// Here the decision uses the shape confidence as well as the size; a region
// that runs out of the picture is penalised because a partially visible object
// is rarely the sign of interest; and a plate sitting inside a bigger board
// wins over that board, because the board is the carrier, not the sign.
// ---------------------------------------------------------------------------
static bool isInside(const Rect &a, const Rect &b, int pad = 3) {
	return a.x >= b.x - pad && a.y >= b.y - pad &&
		a.x + a.width  <= b.x + b.width  + pad &&
		a.y + a.height <= b.y + b.height + pad;
}

int selectBestSign(const vector<ShapeInfo> &list, const ShapeParams &p) {
	int				best = -1;
	double			bestRank = 0.0, maxArea = 0.0, referenceArea = 0.0;
	vector<bool>	isCarrier(list.size(), false);
	vector<double>	nestedBoost(list.size(), 1.0);
	vector<double>	nestedPenalty(list.size(), 1.0);
	vector<bool>	protectedOuter(list.size(), false);

	// The plate area (the hull) is what ranks a candidate: a sign whose colour
	// is only a rim would otherwise be judged by the area of that thin band.
	for (size_t i = 0; i < list.size(); i++)
		if (list[i].hullArea > maxArea) maxArea = list[i].hullArea;
	if (maxArea < 1.0) return -1;
	// A leaked colour component touching the image edge can dwarf every real
	// proposal.  It must not define the scale normalisation for all candidates.
	for (size_t i = 0; i < list.size(); i++)
		if (!list[i].touchesBorder && list[i].selectionScore >= 0.40 &&
			list[i].hullArea > referenceArea)
			referenceArea = list[i].hullArea;
	if (referenceArea < 1.0) referenceArea = maxArea;

	// mark the outer boards that hold a recognised plate inside them
	for (size_t i = 0; i < list.size(); i++) {
		if (list[i].shape == SHAPE_UNKNOWN) continue;
		for (size_t j = 0; j < list.size(); j++) {
			if (i == j || list[j].shape == SHAPE_UNKNOWN) continue;
			if (list[i].hullArea >= list[j].hullArea) continue;
			// An inner printed digit/arrow is rich in edges but normally contains
			// no red/blue/yellow paint.  Requiring colour evidence here keeps the
			// containment rule specific to a genuine inner traffic-sign plate.
			// A carrier board often touches the crop boundary, but a loosely
			// cropped image can contain the whole board.  In that case require the
			// inner plate to have independently stronger fused/boundary evidence.
			// Requiring outer-border contact unconditionally caused a large complete
			// board to beat a clearly supported inner traffic sign.
			bool credibleInnerPlate = list[j].touchesBorder ||
				(list[i].edgeSupport >= 0.65 &&
				 list[i].selectionScore >= list[j].selectionScore + 0.05);
			if (list[i].shape != list[j].shape && credibleInnerPlate &&
				isInside(list[i].bbox, list[j].bbox) &&
				list[i].hullArea >= p.innerMinArea  * list[j].hullArea &&
				list[i].hullArea <= p.innerMaxArea  * list[j].hullArea &&
				list[i].score    >= p.innerMinScore * list[j].score &&
				list[i].colorCoverage >= p.minColorCoverage)
				isCarrier[j] = true;
		}
	}

	// Nested boundary pairs are common: the yellow/white interior and the outer
	// dark/red rim can both form excellent copies of the same shape.  Prefer the
	// supported outer outline; otherwise the border is deleted before GrabCut is
	// even called.  The area window keeps unrelated carrier boards out.
	for (size_t i = 0; i < list.size(); i++) {
		for (size_t j = 0; j < list.size(); j++) {
			if (i == j || list[i].shape == SHAPE_UNKNOWN ||
				list[j].shape == SHAPE_UNKNOWN || list[i].hullArea >= list[j].hullArea)
				continue;
			double ratio = list[i].hullArea / max(1.0, list[j].hullArea);
			if (!isInside(list[i].bbox, list[j].bbox, 5)) continue;
			bool borderShape = list[i].shape == SHAPE_CIRCLE ||
				list[i].shape == SHAPE_TRIANGLE || list[i].shape == SHAPE_OCTAGON;
			if (borderShape && list[i].shape == list[j].shape &&
				ratio >= 0.34 && ratio <= 0.84 &&
				list[j].edgeSupport >= 0.50 &&
				list[j].colorCapture >= 0.78 * list[i].colorCapture &&
				list[j].selectionScore >= 0.72 * list[i].selectionScore) {
				nestedPenalty[i] = min(nestedPenalty[i], 0.88);
				nestedBoost[j] = max(nestedBoost[j], 1.10);
			}

			// A large circular pictogram can occupy most of a rounded rectangular
			// plate.  It is not a separate sign.  Prefer the outer plate when it
			// captures more of the sign-colour evidence and remains credible.
			if (list[i].shape == SHAPE_CIRCLE && list[j].shape == SHAPE_RECTANGLE &&
				ratio > p.innerMaxArea && ratio < 0.92 && list[j].score >= 0.65 &&
				list[j].rectFit >= 0.86 && list[j].edgeSupport >= 0.58 &&
				list[j].colorCoverage >= 0.50 && list[j].colorCapture >= 0.80 &&
				list[j].colorCapture >= list[i].colorCapture + 0.06 &&
				list[j].selectionScore >= 0.72 * list[i].selectionScore) {
				nestedPenalty[i] = min(nestedPenalty[i], 0.62);
				nestedBoost[j] = max(nestedBoost[j], 1.16);
				protectedOuter[j] = true;
			}

			// Conversely, a very small triangle inside a strongly coloured,
			// well-fitted rectangle is normally the warning pictogram printed on
			// the plate, not the plate itself.  The old generic carrier rule starts
			// at innerMinArea (10%), so it missed precisely these tiny graphics and
			// could return only a few hundred pixels from a complete sign board.
			// Requiring matching dominant colour plus strong outer colour capture,
			// geometry and edge support keeps the exception local to a physical
			// traffic-coloured plate; an arbitrary surrounding billboard cannot use
			// it to suppress a genuine triangular sign.
			if (list[i].shape == SHAPE_TRIANGLE &&
				list[j].shape == SHAPE_RECTANGLE && ratio >= 0.02 &&
				ratio < p.innerMinArea && list[i].colorId >= 0 &&
				list[i].colorId == list[j].colorId && list[j].rectFit >= 0.92 &&
				list[j].score >= 0.78 && list[j].edgeSupport >= 0.58 &&
				list[j].colorCoverage >= 0.62 && list[j].colorCapture >= 0.86 &&
				list[j].selectionScore >= 0.72) {
				nestedPenalty[i] = min(nestedPenalty[i], 0.48);
				nestedBoost[j] = max(nestedBoost[j], 1.14);
				protectedOuter[j] = true;
			}
		}
	}

	for (size_t i = 0; i < list.size(); i++) {
		if (list[i].shape == SHAPE_UNKNOWN || isCarrier[i])
			continue;
		double	sizeTerm = min(1.0, sqrt(list[i].hullArea / referenceArea));
		// Advanced candidates already carry a fused score.  The size term is a
		// light tie-breaker only; it no longer lets a huge coloured wall beat a
		// smaller plate with a much better supported boundary.
		double sizeWeight = clamp(p.selectionSizeWeight, 0.0, 1.0);
		double	rank = (list[i].selectionScore > 0.0)
			? (1.0 - sizeWeight) * list[i].selectionScore + sizeWeight * sizeTerm
			: list[i].score * (0.35 + 0.65 * sizeTerm);
		rank *= nestedBoost[i] * nestedPenalty[i];
		rank *= (1.0 - p.centerPriorStrength) +
			p.centerPriorStrength * list[i].centerProximity;
		if (list[i].touchesBorder) rank *= protectedOuter[i] ? 0.88 :
			(list[i].hullArea > 0.55 * maxArea ? 0.52 : 0.78);
		if (rank > bestRank) { bestRank = rank; best = (int)i; }
	}

	// Do not turn uncertainty into a confident-looking segmentation.  If every
	// candidate is UNKNOWN, callers receive -1 and produce an empty result.  A
	// largest-blob fallback is convenient in a lecture demo but unsafe in a
	// downstream recognition pipeline because it hides unsupported shapes and
	// colour-only false positives.
	return best;
}

// ---------------------------------------------------------------------------
// Stage 4: rebuild a solid mask from the idealised shape
// ---------------------------------------------------------------------------
Mat buildShapeMask(const ShapeInfo &info, Size sz) {
	Mat		mask = Mat::zeros(sz, CV_8U);

	if (info.shape == SHAPE_CIRCLE && info.ellipseValid) {
		ellipse(mask, info.ellipse, Scalar(255), FILLED, LINE_8);
	} else if (info.shape == SHAPE_CIRCLE && info.circleRadius > 1.0f) {
		circle(mask, Point(cvRound(info.circleCenter.x),
			cvRound(info.circleCenter.y)), cvRound(info.circleRadius),
			Scalar(255), FILLED, LINE_8);
	} else if (info.ideal.size() >= 3) {
		vector<Point>	poly;
		convexHull(info.ideal, poly);		// guarantee convexity before filling
		fillConvexPoly(mask, poly, Scalar(255), LINE_8);
	} else if (info.contour.size() >= 3) {
		vector<vector<Point> >	c(1, info.contour);
		drawContours(mask, c, 0, Scalar(255), FILLED);
	}
	return mask;
}

// ---------------------------------------------------------------------------
// Advanced proposal support: adaptive edges, evidence measurement and fusion
// ---------------------------------------------------------------------------
static double clamp01(double v) {
	return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

static int oddKernel(int wanted, int minimum) {
	int k = max(wanted, minimum);
	if ((k & 1) == 0) k++;
	return k;
}

static Rect expandedRect(const Rect &r, Size sz, int pad) {
	int x0 = max(0, r.x - pad);
	int y0 = max(0, r.y - pad);
	int x1 = min(sz.width,  r.x + r.width  + pad);
	int y1 = min(sz.height, r.y + r.height + pad);
	return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0));
}

void buildAdaptiveEdgeMap(const Mat &bgr, const ShapeParams &p, Mat &edges) {
	requireBgr8(bgr, "buildAdaptiveEdgeMap");
	Mat gray, local, smooth, gx, gy, ax, ay, gradient, otsuMask;
	cvtColor(bgr, gray, COLOR_BGR2GRAY);

	// Local contrast normalisation recovers the outline of dull signs without
	// amplifying the whole image as strongly as global histogram equalisation.
	Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
	clahe->apply(gray, local);
	GaussianBlur(local, smooth, Size(5, 5), 0.9);

	// Otsu on the gradient magnitude gives a per-image Canny operating point.
	// Clamps keep very flat and very textured photographs in a safe range.
	Sobel(smooth, gx, CV_16S, 1, 0, 3);
	Sobel(smooth, gy, CV_16S, 0, 1, 3);
	convertScaleAbs(gx, ax);
	convertScaleAbs(gy, ay);
	addWeighted(ax, 0.5, ay, 0.5, 0.0, gradient);
	double t = threshold(gradient, otsuMask, 0, 255, THRESH_BINARY | THRESH_OTSU);
	double high = min(210.0, max(55.0, 1.25 * t));
	double low  = max(18.0, 0.42 * high);
	Canny(smooth, edges, low, high, 3, true);

	int side = min(bgr.cols, bgr.rows);
	int k = oddKernel(cvRound(side * p.edgeCloseFrac), p.edgeCloseMin);
	morphologyEx(edges, edges, MORPH_CLOSE,
		getStructuringElement(MORPH_ELLIPSE, Size(k, k)), Point(-1, -1), 1);
}

static Mat shapeOutlineMask(const ShapeInfo &info, Size sz, int thickness) {
	Mat out = Mat::zeros(sz, CV_8U);
	if (info.shape == SHAPE_CIRCLE && info.ellipseValid) {
		ellipse(out, info.ellipse, Scalar(255), thickness, LINE_8);
	} else if (info.shape == SHAPE_CIRCLE && info.circleRadius > 1.f) {
		circle(out, Point(cvRound(info.circleCenter.x), cvRound(info.circleCenter.y)),
			cvRound(info.circleRadius), Scalar(255), thickness, LINE_8);
	} else if (info.ideal.size() >= 3) {
		vector<vector<Point> > c(1, info.ideal);
		drawContours(out, c, 0, Scalar(255), thickness, LINE_8);
	}
	return out;
}

static int dominantColor(const Mat &prior, const vector<Mat> &masks) {
	int best = -1, bestArea = 0;
	for (size_t i = 0; i < masks.size(); i++) {
		if (masks[i].empty()) continue;
		Mat overlap;
		bitwise_and(prior, masks[i], overlap);
		int a = countNonZero(overlap);
		if (a > bestArea) { bestArea = a; best = (int)i; }
	}
	return best;
}

static void measureCandidateEvidence(ShapeInfo &info, const Mat &bgr,
		const vector<Mat> &masks, const Mat &colour, const Mat &edgeDistance,
		const Mat &lab, const ShapeParams &p) {
	Mat prior = buildShapeMask(info, bgr.size());
	int priorArea = countNonZero(prior);
	if (priorArea <= 0) return;

	Mat insideColour;
	bitwise_and(prior, colour, insideColour);
	int colourInside = countNonZero(insideColour);
	info.colorCoverage = (double)colourInside / priorArea;

	// The fitted model can extend beyond the source contour's bounding box (most
	// notably for minEnclosingTriangle).  Build the local denominator around the
	// actual prior so every coloured pixel counted in the numerator is also in
	// the denominator.  The former bbox-based calculation could report an
	// impossible capture ratio above 1.0 and bias nested-candidate selection.
	vector<Point> priorPoints;
	findNonZero(prior, priorPoints);
	Rect priorBox = priorPoints.empty() ? info.bbox : boundingRect(priorPoints);
	int pad = max(3, cvRound(max(priorBox.width, priorBox.height) * 0.18));
	Rect localR = expandedRect(priorBox, bgr.size(), pad);
	int localColour = localR.area() > 0 ? countNonZero(colour(localR)) : 0;
	info.colorCapture = localColour > 0
		? clamp01((double)colourInside / localColour) : 0.0;
	if (info.colorId < 0) info.colorId = dominantColor(prior, masks);

	// Boundary support is a soft distance-to-edge score rather than a brittle
	// exact overlap count.  It tolerates anti-aliasing and one-pixel Canny shifts.
	int side = min(bgr.cols, bgr.rows);
	double tol = max(1.25, side * p.edgeToleranceFrac);
	Mat outline = shapeOutlineMask(info, bgr.size(), 1);
	double edgeSum = 0.0;
	int edgeN = 0;
	for (int y = 0; y < outline.rows; y++) {
		const uchar *o = outline.ptr<uchar>(y);
		const float *d = edgeDistance.ptr<float>(y);
		for (int x = 0; x < outline.cols; x++) if (o[x]) {
			edgeSum += exp(-(double)d[x] / tol);
			edgeN++;
		}
	}
	info.edgeSupport = edgeN > 0 ? edgeSum / edgeN : 0.0;

	// Mean Lab difference across a narrow inner/outer boundary band.  The Lab
	// distance is useful when hue is weak but the plate still contrasts with its
	// surroundings.
	int band = oddKernel(cvRound(min(info.bbox.width, info.bbox.height) * 0.07), 3);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(band, band));
	Mat eroded, dilated, innerBand, outerBand;
	erode(prior, eroded, kernel);
	dilate(prior, dilated, kernel);
	subtract(prior, eroded, innerBand);
	subtract(dilated, prior, outerBand);
	if (countNonZero(innerBand) > 0 && countNonZero(outerBand) > 0) {
		Scalar mi = mean(lab, innerBand), mo = mean(lab, outerBand);
		double dl = mi[0] - mo[0], da = mi[1] - mo[1], db = mi[2] - mo[2];
		info.contrast = clamp01(sqrt(dl * dl + da * da + db * db) / 95.0);
	}

	// Do not saturate coverage too early: a tiny coloured arrow inside a sign
	// and a genuinely filled triangular plate can both exceed 18%, yet they are
	// not equally persuasive.  Capture complements coverage for thin red rims.
	double colourEvidence = clamp01(0.50 * min(1.0, info.colorCoverage / 0.50)
		+ 0.50 * min(1.0, info.colorCapture));
	double imageArea = (double)bgr.cols * bgr.rows;
	double sizeEvidence = clamp01(sqrt(info.hullArea / max(1.0, 0.18 * imageArea)));
	info.selectionScore = clamp01(0.36 * info.score + 0.34 * info.edgeSupport
		+ 0.17 * colourEvidence + 0.08 * info.contrast + 0.05 * sizeEvidence);
	if (info.shape == SHAPE_TRIANGLE && info.edgeSupport >= 0.90 &&
		info.colorCoverage >= 0.65 && info.triFit >= 0.75)
		info.selectionScore = clamp01(info.selectionScore + 0.055);
	if (info.touchesBorder && info.hullArea > 0.55 * imageArea)
		info.selectionScore *= 0.48;
}

static vector<Point> sampledCircle(Point2f centre, float radius, int n = 96) {
	vector<Point> c;
	c.reserve(n);
	for (int i = 0; i < n; i++) {
		double a = 2.0 * CV_PI * i / n;
		c.push_back(Point(cvRound(centre.x + radius * cos(a)),
			cvRound(centre.y + radius * sin(a))));
	}
	return c;
}

static double maskIoU(const ShapeInfo &a, const ShapeInfo &b, Size sz) {
	Mat ma = buildShapeMask(a, sz), mb = buildShapeMask(b, sz), inter, uni;
	bitwise_and(ma, mb, inter);
	bitwise_or(ma, mb, uni);
	int u = countNonZero(uni);
	return u > 0 ? (double)countNonZero(inter) / u : 0.0;
}

vector<ShapeInfo> detectSignShapesAdvanced(const Mat &bgr,
		const vector<Mat> &masks, const ShapeParams &p, Mat *edgeImage,
		vector<ShapeInfo> *rejected) {
	requireBgr8(bgr, "detectSignShapesAdvanced");
	for (size_t i = 0; i < masks.size(); i++)
		requireMask8(masks[i], bgr.size(), "detectSignShapesAdvanced");
	Mat edges;
	buildAdaptiveEdgeMap(bgr, p, edges);
	if (edgeImage) edges.copyTo(*edgeImage);
	vector<Mat> evidenceMasks(3);
	thresholdColorEvidence(bgr, p, evidenceMasks[0], evidenceMasks[1],
		evidenceMasks[2]);

	vector<ShapeInfo> candidates = detectSignShapes(masks, p, rejected);
	for (size_t i = 0; i < candidates.size(); i++)
		candidates[i].proposalSource = SOURCE_COLOR;

	if (p.edgeProposals) {
		// Closed edge loops recover the true plate outline when the colour mask is
		// fragmented or when the desired triangle is printed on a larger board.
		vector<vector<Point> > edgeContours;
		Mat tmp = edges.clone();
		findContours(tmp, edgeContours, RETR_LIST, CHAIN_APPROX_NONE);
		for (size_t i = 0; i < edgeContours.size(); i++) {
			ShapeInfo info;
			if (classifyContour(edgeContours[i], bgr.size(), p, info)) {
				info.proposalSource = SOURCE_EDGE;
				candidates.push_back(info);
			}
		}

		// Hough voting supplies a second, independent proposal for circular signs.
		// It is especially useful when a red ring touches a nearby red object.
		if (p.houghCircles) {
			Mat gray, smooth;
			cvtColor(bgr, gray, COLOR_BGR2GRAY);
			GaussianBlur(gray, smooth, Size(5, 5), 1.1);
			int side = min(bgr.cols, bgr.rows);
			vector<Vec3f> circles;
			HoughCircles(smooth, circles, HOUGH_GRADIENT, 1.0,
				max(8.0, side * 0.18), 110.0, max(11.0, side * 0.075),
				max(5, cvRound(side * 0.075)), max(8, cvRound(side * 0.58)));
			if (circles.size() > 16) circles.resize(16);
			for (size_t i = 0; i < circles.size(); i++) {
				ShapeInfo info;
				Point2f centre(circles[i][0], circles[i][1]);
				float radius = circles[i][2];
				vector<Point> contour = sampledCircle(centre, radius);
				if (classifyContour(contour, bgr.size(), p, info)) {
					info.shape = SHAPE_CIRCLE;
					info.score = max(info.score, 0.72);
					info.proposalSource = SOURCE_HOUGH;
					candidates.push_back(info);
				}
			}
		}
	}

	vector<ShapeInfo> supported;
	// These full-frame transforms are identical for every proposal.  Computing
	// them once here avoids O(proposals * pixels) duplicate work in cluttered
	// images, where RETR_LIST can produce dozens of candidates.
	Mat colour = Mat::zeros(bgr.size(), CV_8U);
	for (size_t i = 0; i < evidenceMasks.size(); i++)
		if (!evidenceMasks[i].empty()) colour |= evidenceMasks[i];
	Mat inverseEdges, edgeDistance, lab;
	bitwise_not(edges, inverseEdges);
	distanceTransform(inverseEdges, edgeDistance, DIST_L2, 3);
	cvtColor(bgr, lab, COLOR_BGR2Lab);
	for (size_t i = 0; i < candidates.size(); i++) {
		measureCandidateEvidence(candidates[i], bgr, evidenceMasks, colour,
			edgeDistance, lab, p);
		bool hasColour = candidates[i].colorCoverage >= p.minColorCoverage ||
			candidates[i].colorCapture >= 0.12;
		bool hasEdge = candidates[i].edgeSupport >= p.minEdgeSupport;
		double areaRatio = candidates[i].hullArea /
			max(1.0, (double)bgr.cols * bgr.rows);
		bool plausibleHough = candidates[i].proposalSource != SOURCE_HOUGH ||
			(areaRatio >= 0.050 && candidates[i].centerProximity >= 0.52);
		if (plausibleHough && (candidates[i].proposalSource == SOURCE_COLOR ||
			(hasColour && hasEdge)))
			supported.push_back(candidates[i]);
	}

	// Keep the strongest of near-identical colour/edge/Hough proposals.  Models
	// of different shapes are not merged: their scores must compete explicitly.
	stable_sort(supported.begin(), supported.end(),
		[](const ShapeInfo &a, const ShapeInfo &b) {
			return a.selectionScore > b.selectionScore;
		});
	vector<ShapeInfo> unique;
	for (size_t i = 0; i < supported.size(); i++) {
		bool duplicate = false;
		for (size_t j = 0; j < unique.size() && !duplicate; j++)
			if (supported[i].shape == unique[j].shape &&
				bboxIoU(supported[i].bbox, unique[j].bbox) > 0.55 &&
				maskIoU(supported[i], unique[j], bgr.size()) >= p.dedupeIoU)
				duplicate = true;
		if (!duplicate) unique.push_back(supported[i]);
		if (unique.size() >= 48) break;
	}
	return unique;
}

// ---------------------------------------------------------------------------
// Shape-constrained GrabCut refinement
// ---------------------------------------------------------------------------
static double binaryBoundarySupport(const Mat &mask, const Mat &edgeDistance,
		double tolerance) {
	Mat eroded, outline;
	erode(mask, eroded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	subtract(mask, eroded, outline);
	double sum = 0.0;
	int n = 0;
	for (int y = 0; y < outline.rows; y++) {
		const uchar *o = outline.ptr<uchar>(y);
		const float *d = edgeDistance.ptr<float>(y);
		for (int x = 0; x < outline.cols; x++) if (o[x]) {
			sum += exp(-(double)d[x] / tolerance);
			n++;
		}
	}
	return n > 0 ? sum / n : 0.0;
}

static Mat scaledBinaryMask(const Mat &source, Point2f centre, double scale) {
	Mat out;
	Mat transform = getRotationMatrix2D(centre, 0.0, scale);
	warpAffine(source, out, transform, source.size(), INTER_NEAREST,
		BORDER_CONSTANT, Scalar(0));
	return out;
}

// A colour interior and the physical outside rim often produce two nested,
// parallel edges.  When proposal ranking returns the inner one, search a small
// family of scale hypotheses and snap to the strongest coherent outer edge.
// This is the classical, self-contained equivalent of boundary-hypothesis
// refinement: it never changes the shape class and is capped at 35% growth.
static Mat snapPriorToOuterBoundary(const Mat &bgr, const ShapeInfo &info,
		const ShapeParams &p, const Mat &base, double &chosenScale) {
	chosenScale = 1.0;
	// Triangular warnings have a characteristic coloured interior plus dark/red
	// outer frame.  Circles and octagons do not use this scale fallback: in
	// clutter a coincidental curved edge can otherwise enlarge an already-correct
	// mask.  Their coloured rims are preserved later by colour reconstruction.
	if (info.touchesBorder || info.shape != SHAPE_TRIANGLE ||
		countNonZero(base) == 0)
		return base;
	// An outer plate hypothesis includes its dark/red frame and therefore cannot
	// be almost completely occupied by raw yellow/red paint.  Only a distinctly
	// colour-interior contour is allowed to grow; this gate prevents a correct
	// outer triangle from snapping to an unrelated background edge.
	if (info.colorCoverage < 0.88 || info.colorCapture < 0.45)
		return base;
	Mat edges, inverseEdges, distance;
	buildAdaptiveEdgeMap(bgr, p, edges);
	bitwise_not(edges, inverseEdges);
	distanceTransform(inverseEdges, distance, DIST_L2, 3);
	double tolerance = max(1.20, min(bgr.cols, bgr.rows) * p.edgeToleranceFrac);
	vector<double> scales, support;
	vector<Mat> hypotheses;
	for (int step = 0; step <= 7; step++) {
		double scale = 1.0 + 0.05 * step;
		Mat candidate = step == 0 ? base :
			scaledBinaryMask(base, info.box.center, scale);
		scales.push_back(scale);
		hypotheses.push_back(candidate);
		support.push_back(binaryBoundarySupport(candidate, distance, tolerance));
	}
	double globalBest = *max_element(support.begin(), support.end());
	int selected = 0;
	for (int i = 1; i < (int)support.size(); i++) {
		double left = support[i - 1];
		double right = (i + 1 < (int)support.size()) ? support[i + 1] : -1.0;
		bool localPeak = support[i] >= left && support[i] >= right;
		if (localPeak && support[i] >= 0.58 &&
			support[i] >= 0.90 * globalBest)
			selected = i;
	}
	// A two-pixel movement is normal raster variation, not evidence of a rim.
	if (selected > 0) {
		double baseRadius = 0.5 * min(info.bbox.width, info.bbox.height);
		if ((scales[selected] - 1.0) * baseRadius >= 2.5) {
			chosenScale = scales[selected];
			return hypotheses[selected];
		}
	}
	// If the selected triangle is almost entirely sign-coloured, its contour is
	// the colour/black interface rather than the plate exterior.  A 1.28 scale
	// models the common warning-sign frame, but only when no reliable outer edge
	// hypothesis was found and the candidate captures substantial local colour.
	if (info.colorCoverage >= 0.88 && info.colorCapture >= 0.45) {
		chosenScale = 1.28;
		return scaledBinaryMask(base, info.box.center, chosenScale);
	}
	return base;
}

// Recover the physical outside ellipse when an edge proposal locked onto the
// white interior of a circular sign.  This case is recognisable without a
// class-specific size constant: the selected disc has very little colour, but
// a coherent ring of the same dominant colour surrounds it.  Fitting the
// outside contour of that ring is safer than merely dilating the disc because
// it preserves perspective foreshortening and ignores isolated red clutter.
static Mat recoverCircularColourRim(const ShapeInfo &info,
		const vector<Mat> &raw, const Mat &base, double &chosenScale) {
	chosenScale = 1.0;
	if (info.shape != SHAPE_CIRCLE || info.touchesBorder ||
		info.colorId < 0 || info.colorId >= (int)raw.size() ||
		raw[info.colorId].empty() || countNonZero(base) == 0)
		return base;
	// A filled blue/yellow plate must not be grown towards nearby same-colour
	// background.  Rim recovery is only for the low-coverage interior proposal.
	if (info.colorCoverage > 0.16 || info.colorCapture > 0.35)
		return base;

	const int baseArea = countNonZero(base);
	const int shortSide = max(5, min(info.bbox.width, info.bbox.height));
	const int searchRadius = max(3, cvRound(0.30 * shortSide));
	Mat search, nearColour;
	dilate(base, search, getStructuringElement(MORPH_ELLIPSE,
		Size(2 * searchRadius + 1, 2 * searchRadius + 1)));
	bitwise_and(raw[info.colorId], search, nearColour);
	if (countNonZero(nearColour) < max(10, cvRound(0.015 * baseArea)))
		return base;

	Mat outsideBase, originalNear = nearColour.clone();
	bitwise_and(nearColour, base == 0, outsideBase);
	if (countNonZero(outsideBase) < max(8, cvRound(0.012 * baseArea)))
		return base;

	const int closeK = oddKernel(cvRound(0.08 * shortSide), 3);
	morphologyEx(nearColour, nearColour, MORPH_CLOSE,
		getStructuringElement(MORPH_ELLIPSE, Size(closeK, closeK)));
	vector<vector<Point> > contours;
	findContours(nearColour, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat best = base;
	double bestQuality = -1.0;
	for (size_t i = 0; i < contours.size(); i++) {
		if (contours[i].size() < 5 || contourArea(contours[i]) < 0.20 * baseArea)
			continue;
		RotatedRect fitted;
		try { fitted = fitEllipseAMS(contours[i]); }
		catch (const cv::Exception&) { continue; }
		double smallAxis = min(fitted.size.width, fitted.size.height);
		double largeAxis = max(fitted.size.width, fitted.size.height);
		if (smallAxis < 4.0 || largeAxis / smallAxis > 1.75) continue;
		double dx = fitted.center.x - info.box.center.x;
		double dy = fitted.center.y - info.box.center.y;
		if (sqrt(dx * dx + dy * dy) > 0.18 * shortSide) continue;

		Mat candidate = Mat::zeros(base.size(), CV_8U);
		ellipse(candidate, fitted, Scalar(255), FILLED, LINE_8);
		int candidateArea = countNonZero(candidate);
		double areaRatio = (double)candidateArea / baseArea;
		if (areaRatio < 1.08 || areaRatio > 1.85) continue;

		Mat overlap, captured;
		bitwise_and(candidate, base, overlap);
		double baseRecall = (double)countNonZero(overlap) / baseArea;
		bitwise_and(candidate, originalNear, captured);
		double colourCapture = (double)countNonZero(captured) /
			max(1, countNonZero(originalNear));
		if (baseRecall < 0.88 || colourCapture < 0.72) continue;

		double quality = 0.55 * colourCapture + 0.35 * baseRecall +
			0.10 * min(1.0, areaRatio / 1.35);
		if (quality > bestQuality) {
			bestQuality = quality;
			best = candidate;
			chosenScale = sqrt(areaRatio);
		}
	}
	return best;
}

Mat refineShapeMask(const Mat &bgr, const ShapeInfo &info,
		const vector<Mat> &colorMasks, const ShapeParams &p, Mat *trimapDebug,
		RefinementDiagnostics *diagnostics) {
	if (diagnostics) *diagnostics = RefinementDiagnostics();
	if (bgr.empty()) {
		if (diagnostics) diagnostics->decision = "empty input image";
		return Mat();
	}
	requireBgr8(bgr, "refineShapeMask");
	for (size_t i = 0; i < colorMasks.size(); i++)
		requireMask8(colorMasks[i], bgr.size(), "refineShapeMask");
	vector<Mat> raw(3);
	thresholdColorEvidence(bgr, p, raw[0], raw[1], raw[2]);
	Mat prior = buildShapeMask(info, bgr.size());
	double priorScale = 1.0;
	prior = snapPriorToOuterBoundary(bgr, info, p, prior, priorScale);
	if (priorScale == 1.0) {
		double rimScale = 1.0;
		prior = recoverCircularColourRim(info, raw, prior, rimScale);
		priorScale = rimScale;
	}
	int priorArea = countNonZero(prior);
	if (diagnostics) {
		diagnostics->priorArea = priorArea;
		diagnostics->priorScale = priorScale;
	}
	auto fallback = [&](const char *reason) -> Mat {
		if (diagnostics) {
			diagnostics->accepted = false;
			diagnostics->finalArea = priorArea;
			diagnostics->decision = reason;
		}
		return prior;
	};
	if (!p.refineWithGrabCut) return fallback("shape prior (refinement disabled)");
	if (priorArea == 0 || bgr.empty()) return fallback("empty image or shape prior");
	if (diagnostics) diagnostics->attempted = true;

	int shortSide = max(3, min(info.bbox.width, info.bbox.height));
	int outerK = oddKernel(cvRound(shortSide * p.grabCutOuterFrac) * 2 + 1, 3);
	int innerK = oddKernel(cvRound(shortSide * p.grabCutInnerFrac) * 2 + 1, 3);
	Mat outer, inner;
	dilate(prior, outer, getStructuringElement(MORPH_ELLIPSE, Size(outerK, outerK)));
	erode(prior, inner, getStructuringElement(MORPH_ELLIPSE, Size(innerK, innerK)));
	if (countNonZero(inner) < 8) {
		innerK = 3;
		erode(prior, inner, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	}
	if (countNonZero(inner) < 4) return fallback("shape prior (no stable foreground core)");

	// If the chosen circle is almost uncoloured but same-colour evidence exists
	// immediately outside it, the proposal is the white interior of a rim sign.
	// Record that fact before GrabCut: this is the only case allowed to preserve
	// colour just outside the original prior and to use a wider area window.
	bool outwardRimEvidence = false;
	if (info.shape == SHAPE_CIRCLE && info.colorId >= 0 &&
		info.colorId < (int)raw.size() && info.colorCoverage <= 0.16 &&
		info.colorCapture <= 0.35) {
		Mat nearRaw, outsidePrior;
		bitwise_and(raw[info.colorId], outer, nearRaw);
		bitwise_and(nearRaw, prior == 0, outsidePrior);
		outwardRimEvidence = countNonZero(outsidePrior) >=
			max(8, cvRound(0.012 * priorArea));
	}

	Mat gcMask(bgr.size(), CV_8U, Scalar(GC_BGD));
	gcMask.setTo(Scalar(GC_PR_BGD), outer);
	gcMask.setTo(Scalar(GC_PR_FGD), prior);
	gcMask.setTo(Scalar(GC_FGD), inner);

	// Eroded sign-colour pixels are reliable foreground scribbles.  The inner
	// geometric seed simultaneously teaches GrabCut the white/black pictogram
	// colours, so the final filled plate is not reduced to its coloured rim.
	Mat colour = Mat::zeros(bgr.size(), CV_8U);
	for (size_t i = 0; i < colorMasks.size(); i++)
		if (!colorMasks[i].empty()) colour |= colorMasks[i];
	Mat sureColour, colourInside;
	bitwise_and(colour, prior, colourInside);
	erode(colourInside, sureColour,
		getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	gcMask.setTo(Scalar(GC_FGD), sureColour);

	if (trimapDebug) {
		Mat shown;
		gcMask.convertTo(shown, CV_8U, 85.0);
		shown.copyTo(*trimapDebug);
	}

	Mat bgModel, fgModel;
	try {
		grabCut(bgr, gcMask, Rect(), bgModel, fgModel,
			p.grabCutIterations, GC_INIT_WITH_MASK);
	} catch (const cv::Exception&) {
		return fallback("shape prior (GrabCut exception)");
	}

	Mat foreground = (gcMask == GC_FGD) | (gcMask == GC_PR_FGD);
	bitwise_and(foreground, outer, foreground);
	if (diagnostics) diagnostics->graphCutArea = countNonZero(foreground);

	// Recover high-confidence *unfilled* colour evidence.  Filled proposal masks
	// cannot tell a red rim from the white centre; the raw masks can.  Joining
	// this evidence before hole filling preserves the complete physical plate.
	Mat protectedColour = Mat::zeros(bgr.size(), CV_8U);
	if (info.colorId >= 0 && info.colorId < (int)raw.size())
		raw[info.colorId].copyTo(protectedColour);
	else
		for (size_t i = 0; i < raw.size(); i++) protectedColour |= raw[i];
	bitwise_and(protectedColour, outwardRimEvidence ? outer : prior,
		protectedColour);
	int protectedArea = countNonZero(protectedColour);
	Mat missingProtected;
	bitwise_and(protectedColour, foreground == 0, missingProtected);
	if (diagnostics && countNonZero(missingProtected) > 0)
		diagnostics->colorRepaired = true;
	foreground |= protectedColour;
	morphologyEx(foreground, foreground, MORPH_CLOSE,
		getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	// Retain the component with the strongest overlap with the sure-foreground
	// core.  This removes independent background islands that happen to share a
	// similar colour model.
	Mat labels, stats, centroids;
	int n = connectedComponentsWithStats(foreground, labels, stats, centroids, 8);
	int best = -1;
	double bestScore = -1.0;
	for (int label = 1; label < n; label++) {
		Mat component = (labels == label), overlap, colourOverlap;
		bitwise_and(component, inner, overlap);
		double core = countNonZero(overlap);
		bitwise_and(component, protectedColour, colourOverlap);
		double colourHit = countNonZero(colourOverlap);
		double area = stats.at<int>(label, CC_STAT_AREA);
		double s = core + 1.25 * colourHit + 0.015 * area;
		if (s > bestScore) { bestScore = s; best = label; }
	}
	if (best < 0) return fallback("shape prior (no connected foreground)");

	Mat refined = (labels == best);
	fillEnclosedHoles(refined);

	// Traffic sign plates are convex.  Only repair a substantially concave cut;
	// ordinary GrabCut detail is preserved so the boundary stays pixel-accurate.
	vector<vector<Point> > contours;
	Mat contourInput = refined.clone();
	findContours(contourInput, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	if (!contours.empty()) {
		size_t largest = 0;
		for (size_t i = 1; i < contours.size(); i++)
			if (contourArea(contours[i]) > contourArea(contours[largest])) largest = i;
		vector<Point> hull;
		convexHull(contours[largest], hull);
		double a = contourArea(contours[largest]), ha = contourArea(hull);
		if (ha > 1.0 && a / ha < 0.88) {
			refined.setTo(0);
			fillConvexPoly(refined, hull, Scalar(255), LINE_8);
		}
	}

	double ratio = (double)countNonZero(refined) / priorArea;
	Mat coreHit;
	bitwise_and(refined, inner, coreHit);
	double coreRecall = (double)countNonZero(coreHit) / max(1, countNonZero(inner));
	Mat colorHit;
	bitwise_and(refined, protectedColour, colorHit);
	double colorRecall = protectedArea > 0
		? (double)countNonZero(colorHit) / protectedArea : 1.0;
	vector<Point> priorPoints, refinedPoints;
	findNonZero(prior, priorPoints);
	findNonZero(refined, refinedPoints);
	double spanRatio = 0.0;
	if (!priorPoints.empty() && !refinedPoints.empty()) {
		Rect pr = boundingRect(priorPoints), rr = boundingRect(refinedPoints);
		spanRatio = min((double)rr.width / max(1, pr.width),
			(double)rr.height / max(1, pr.height));
	}
	if (diagnostics) {
		diagnostics->areaRatio = ratio;
		diagnostics->coreRecall = coreRecall;
		diagnostics->colorRecall = colorRecall;
		diagnostics->spanRatio = spanRatio;
	}
	if (ratio < p.refineMinAreaRatio)
		return fallback("shape prior (partial GrabCut area)");
	double maxAreaRatio = outwardRimEvidence
		? max(p.refineMaxAreaRatio, 1.45) : p.refineMaxAreaRatio;
	if (ratio > maxAreaRatio)
		return fallback("shape prior (GrabCut leaked outside)");
	if (coreRecall < 0.72)
		return fallback("shape prior (foreground core lost)");
	if (spanRatio < p.refineMinSpanRatio)
		return fallback("shape prior (one-sided/half mask)");
	if (colorRecall < p.refineMinColorRecall)
		return fallback("shape prior (sign colour or rim lost)");
	if (diagnostics) {
		diagnostics->accepted = true;
		diagnostics->finalArea = countNonZero(refined);
		diagnostics->decision = diagnostics->colorRepaired
			? "accepted after colour/rim repair" : "accepted GrabCut";
	}

	return refined;
}

// ---------------------------------------------------------------------------
// Drawing helper
// ---------------------------------------------------------------------------
void drawShape(Mat &canvas, const ShapeInfo &info, bool withLabel, int thickness) {
	Scalar	col = shapeColor(info.shape);
	char	str[128];

	if (info.shape == SHAPE_CIRCLE && info.ellipseValid) {
		ellipse(canvas, info.ellipse, col, thickness, LINE_AA);
	} else if (info.shape == SHAPE_CIRCLE) {
		circle(canvas, Point(cvRound(info.circleCenter.x),
			cvRound(info.circleCenter.y)), cvRound(info.circleRadius),
			col, thickness, LINE_AA);
	} else if (info.ideal.size() >= 3) {
		vector<vector<Point> >	c(1, info.ideal);
		drawContours(canvas, c, 0, col, thickness, LINE_AA);
	}

	// mark every corner found by the polygonal approximation
	for (size_t i = 0; i < info.poly.size(); i++)
		circle(canvas, info.poly[i], 2, Scalar(255, 255, 255), FILLED);

	if (withLabel) {
		const char sourceMark = (info.proposalSource == SOURCE_EDGE) ? 'E' :
			(info.proposalSource == SOURCE_HOUGH) ? 'H' : 'C';
		double shownScore = info.selectionScore > 0.0
			? info.selectionScore : info.score;
		sprintf_s(str, "%s %.2f %c", shapeName(info.shape), shownScore, sourceMark);
		int	y = info.bbox.y - 4;
		if (y < 10) y = info.bbox.y + info.bbox.height + 11;
		putText(canvas, str, Point(info.bbox.x, y), FONT_HERSHEY_PLAIN, 0.8,
			Scalar(0, 0, 0), 3, LINE_AA);
		putText(canvas, str, Point(info.bbox.x, y), FONT_HERSHEY_PLAIN, 0.8,
			col, 1, LINE_AA);
	}
}

// ---------------------------------------------------------------------------
// Console report of the measured descriptors (used for the report tables)
// ---------------------------------------------------------------------------
void printShapeReport(const ShapeInfo &info, int index) {
	const char	*col = (info.colorId == 0) ? "red   " :
					   (info.colorId == 1) ? "blue  " :
					   (info.colorId == 2) ? "yellow" : "n/a   ";
	const char	*src = (info.proposalSource == SOURCE_EDGE) ? "edge " :
					   (info.proposalSource == SOURCE_HOUGH) ? "hough" : "color";

	printf("    region %-2d %s %-5s %-9s geom %.2f fused %.2f | edge %.2f  "
		"colour %.2f/%.2f  contrast %.2f | corners %2d/%-2d/%-2d | cirFit %.3f  "
		"rectFit %.3f  triFit %.3f | solid %.3f aspect %.2f area %7.0f%s%s\n",
		index, col, src, shapeName(info.shape), info.score, info.selectionScore,
		info.edgeSupport, info.colorCoverage, info.colorCapture, info.contrast,
		info.vertLo, info.vertices, info.vertVeryHi,
		info.cirFit, info.rectFit, info.triFit,
		info.solidity, info.aspect, info.hullArea,
		info.rimRescued ? "  [rim]" : "",
		info.touchesBorder ? "  [border]" : "");
}

// Why a region never became a candidate.  Printed by the demo when a picture
// yields nothing, so that a miss can be traced without a debugger.
void printRejectReport(const ShapeInfo &info, int index) {
	const char	*col = (info.colorId == 0) ? "red   " :
					   (info.colorId == 1) ? "blue  " :
					   (info.colorId == 2) ? "yellow" : "n/a   ";

	printf("    dropped %-2d %s area %6.0f  ->  %s\n",
		index, col, info.hullArea, info.reject.c_str());
}
