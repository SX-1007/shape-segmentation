#include "SegmentationTest.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

MaskMetrics::MaskMetrics()
	: iou(0.0), dice(0.0), precision(0.0), recall(0.0), boundaryIoU(0.0) {}

static Mat binary8(const Mat &input) {
	Mat gray, out;
	if (input.empty()) return out;
	if (input.channels() == 1) gray = input;
	else cvtColor(input, gray, COLOR_BGR2GRAY);
	compare(gray, 0, out, CMP_GT);
	return out;
}

static double safeRatio(double numerator, double denominator, double emptyValue) {
	return denominator > 0.0 ? numerator / denominator : emptyValue;
}

static Mat innerBoundary(const Mat &mask, int width) {
	// Match the authors' Boundary-IoU reference implementation: pad by one so
	// objects truncated at the image edge still have an outer boundary, then
	// apply a 3x3 square erosion `width` times (rather than one large ellipse).
	Mat padded, eroded, boundary;
	copyMakeBorder(mask, padded, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
	erode(padded, eroded, Mat::ones(3, 3, CV_8U), Point(-1, -1),
		max(1, width));
	Mat cropped = eroded(Rect(1, 1, mask.cols, mask.rows));
	subtract(mask, cropped, boundary);
	return boundary;
}

MaskMetrics evaluateBinaryMask(const Mat &prediction, const Mat &groundTruth) {
	MaskMetrics m;
	Mat pred = binary8(prediction), truth = binary8(groundTruth);
	if (pred.empty() || truth.empty() || pred.size() != truth.size()) return m;

	Mat intersection, unionMask;
	bitwise_and(pred, truth, intersection);
	bitwise_or(pred, truth, unionMask);
	double tp = countNonZero(intersection);
	double pa = countNonZero(pred);
	double ga = countNonZero(truth);
	double ua = countNonZero(unionMask);
	m.iou       = safeRatio(tp, ua, 1.0);
	m.dice      = safeRatio(2.0 * tp, pa + ga, 1.0);
	m.precision = safeRatio(tp, pa, ga == 0.0 ? 1.0 : 0.0);
	m.recall    = safeRatio(tp, ga, pa == 0.0 ? 1.0 : 0.0);

	// Boundary-IoU's reference setting uses 2% of the image diagonal.
	double diagonal = sqrt((double)truth.cols * truth.cols +
		(double)truth.rows * truth.rows);
	int boundaryWidth = max(1, cvRound(0.020 * diagonal));
	Mat pb = innerBoundary(pred, boundaryWidth);
	Mat gb = innerBoundary(truth, boundaryWidth);
	bitwise_and(pb, gb, intersection);
	bitwise_or(pb, gb, unionMask);
	m.boundaryIoU = safeRatio(countNonZero(intersection),
		countNonZero(unionMask), 1.0);
	return m;
}

int runShapeDetectionSelfTests() {
	int passed = 0, failed = 0;
	auto check = [&](bool condition, const string &name) {
		cout << "  " << (condition ? "[PASS] " : "[FAIL] ") << name << '\n';
		condition ? passed++ : failed++;
	};
	auto near = [](double a, double b) { return abs(a - b) <= 1e-9; };

	cout << "\nDeterministic shape/segmentation self-tests\n";
	Mat empty = Mat::zeros(64, 64, CV_8U);
	MaskMetrics emptyMetrics = evaluateBinaryMask(empty, empty);
	check(near(emptyMetrics.iou, 1.0) && near(emptyMetrics.boundaryIoU, 1.0),
		"two empty masks score as an exact match");

	Mat borderObject = Mat::zeros(64, 64, CV_8U);
	rectangle(borderObject, Rect(0, 8, 31, 40), Scalar(255), FILLED);
	MaskMetrics exact = evaluateBinaryMask(borderObject, borderObject);
	check(near(exact.iou, 1.0) && near(exact.boundaryIoU, 1.0),
		"Boundary-IoU handles objects truncated by an image border");
	Mat disjoint = Mat::zeros(64, 64, CV_8U);
	rectangle(disjoint, Rect(40, 8, 20, 40), Scalar(255), FILLED);
	MaskMetrics separated = evaluateBinaryMask(borderObject, disjoint);
	check(near(separated.iou, 0.0) && near(separated.boundaryIoU, 0.0),
		"disjoint masks have zero region and boundary overlap");

	ShapeParams p;
	check(detectSignShapes(Mat(), p).empty(),
		"an empty candidate mask is handled without an OpenCV assertion");
	ShapeInfo unknown;
	unknown.hullArea = 1200.0;
	unknown.shape = SHAPE_UNKNOWN;
	check(selectBestSign(vector<ShapeInfo>(1, unknown), p) == -1,
		"an UNKNOWN-only proposal set abstains instead of selecting a blob");
	bool rejectedWrongType = false;
	try {
		Mat gray = Mat::zeros(32, 32, CV_8U), r, b, y;
		buildColorMasks(gray, p, r, b, y);
	} catch (const cv::Exception&) {
		rejectedWrongType = true;
	}
	check(rejectedWrongType, "a non-BGR input fails with an explicit format error");
	Mat ring = Mat::zeros(96, 96, CV_8U);
	circle(ring, Point(48, 48), 30, Scalar(255), 5, LINE_8);
	Mat filled = cleanMask(ring, p);
	check(filled.at<uchar>(48, 48) == 255,
		"morphological cleanup fills enclosed pictogram/rim holes");

	const SignShape expected[4] = { SHAPE_CIRCLE, SHAPE_TRIANGLE,
		SHAPE_RECTANGLE, SHAPE_OCTAGON };
	for (int shapeIndex = 0; shapeIndex < 4; shapeIndex++) {
		Mat mask = Mat::zeros(160, 160, CV_8U);
		if (expected[shapeIndex] == SHAPE_CIRCLE) {
			circle(mask, Point(80, 80), 52, Scalar(255), FILLED, LINE_8);
		} else if (expected[shapeIndex] == SHAPE_TRIANGLE) {
			vector<Point> points = { Point(80, 18), Point(20, 140), Point(140, 140) };
			fillConvexPoly(mask, points, Scalar(255), LINE_8);
		} else if (expected[shapeIndex] == SHAPE_RECTANGLE) {
			rectangle(mask, Rect(28, 42, 104, 76), Scalar(255), FILLED, LINE_8);
		} else {
			vector<Point> points;
			for (int k = 0; k < 8; k++) {
				double a = CV_PI / 8.0 + k * CV_PI / 4.0;
				points.push_back(Point(cvRound(80 + 56 * cos(a)),
					cvRound(80 + 56 * sin(a))));
			}
			fillConvexPoly(mask, points, Scalar(255), LINE_8);
		}
		vector<vector<Point> > contours;
		findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		ShapeInfo info;
		bool classified = !contours.empty() &&
			classifyContour(contours[0], mask.size(), p, info);
		check(classified && info.shape == expected[shapeIndex],
			string("ideal ") + shapeName(expected[shapeIndex]) + " is classified correctly");
		if (classified) {
			Mat rebuilt = buildShapeMask(info, mask.size());
			check(rebuilt.type() == CV_8U && countNonZero(rebuilt) > 0,
				string("ideal ") + shapeName(expected[shapeIndex]) + " rebuilds a non-empty 8-bit mask");
		}
	}

	cout << "Self-test result: " << passed << " passed, " << failed << " failed.\n";
	return failed == 0 ? 0 : 1;
}

static void drawBackground(Mat &image, int scene, RNG &rng) {
	image.create(256, 256, CV_8UC3);
	Scalar top, bottom;
	switch (scene % 8) {
	case 0: top = Scalar(45, 86, 42);  bottom = Scalar(31, 58, 25); break;
	case 1: top = Scalar(80, 105, 142); bottom = Scalar(55, 74, 106); break;
	case 2: top = Scalar(205, 175, 125); bottom = Scalar(105, 100, 92); break;
	case 3: top = Scalar(125, 125, 125); bottom = Scalar(68, 73, 78); break;
	case 4: top = Scalar(75, 64, 54); bottom = Scalar(28, 30, 32); break;
	case 5: top = Scalar(48, 62, 83); bottom = Scalar(92, 109, 130); break;
	case 6: top = Scalar(150, 150, 145); bottom = Scalar(78, 82, 86); break;
	default: top = Scalar(28, 32, 35); bottom = Scalar(8, 10, 14); break;
	}
	for (int y = 0; y < image.rows; y++) {
		double a = (double)y / max(1, image.rows - 1);
		Vec3b c;
		for (int ch = 0; ch < 3; ch++)
			c[ch] = saturate_cast<uchar>((1.0 - a) * top[ch] + a * bottom[ch]);
		image.row(y).setTo(c);
	}

	if (scene % 8 == 0) { // foliage: many natural, irregular green regions
		for (int i = 0; i < 180; i++) {
			Point p(rng.uniform(0, 256), rng.uniform(0, 256));
			int r = rng.uniform(2, 15);
			Scalar c(rng.uniform(20, 75), rng.uniform(55, 145), rng.uniform(18, 70));
			circle(image, p, r, c, FILLED, LINE_AA);
		}
	} else if (scene % 8 == 1) { // brick wall
		for (int y = 12; y < 256; y += 24) line(image, Point(0, y),
			Point(255, y), Scalar(155, 165, 170), 2);
		for (int row = 0, y = 0; y < 256; y += 24, row++)
			for (int x = (row & 1) ? 20 : 0; x < 256; x += 40)
				line(image, Point(x, y), Point(x, min(255, y + 24)),
					Scalar(145, 150, 156), 2);
	} else if (scene % 8 == 2) { // skyline and windows
		for (int x = -10; x < 266; x += rng.uniform(24, 43)) {
			int roof = rng.uniform(95, 180), w = rng.uniform(25, 55);
			rectangle(image, Rect(max(0, x), roof, min(w, 256 - max(0, x)), 256 - roof),
				Scalar(rng.uniform(55, 115), rng.uniform(55, 115), rng.uniform(55, 115)), FILLED);
			for (int wy = roof + 9; wy < 245; wy += 17)
				for (int wx = max(0, x) + 6; wx < min(256, x + w - 4); wx += 13)
					rectangle(image, Rect(wx, wy, 6, 7), Scalar(175, 160, 100), FILLED);
		}
	} else { // urban clutter, including sign-coloured distractors
		for (int i = 0; i < 55; i++) {
			Point a(rng.uniform(-20, 256), rng.uniform(0, 256));
			Point b(a.x + rng.uniform(8, 70), a.y + rng.uniform(4, 45));
			Scalar c;
			if (i % 11 == 0) c = Scalar(35, 35, 185);
			else if (i % 13 == 0) c = Scalar(170, 75, 35);
			else if (i % 17 == 0) c = Scalar(30, 190, 205);
			else c = Scalar(rng.uniform(30, 180), rng.uniform(30, 180), rng.uniform(30, 180));
			rectangle(image, a, b, c, FILLED, LINE_AA);
		}
		for (int i = 0; i < 18; i++)
			line(image, Point(rng.uniform(0, 256), rng.uniform(0, 256)),
				Point(rng.uniform(0, 256), rng.uniform(0, 256)),
				Scalar(rng.uniform(25, 210), rng.uniform(25, 210), rng.uniform(25, 210)),
				rng.uniform(1, 4), LINE_AA);
	}
}

static void drawSyntheticSign(SignShape shape, Mat &sign, Mat &mask) {
	sign = Mat::zeros(160, 160, CV_8UC3);
	mask = Mat::zeros(160, 160, CV_8U);
	const Scalar red(35, 35, 205), blue(190, 82, 28), yellow(35, 205, 225);
	const Scalar white(225, 230, 230), dark(22, 25, 28);
	if (shape == SHAPE_CIRCLE) {
		circle(mask, Point(80, 80), 58, Scalar(255), FILLED, LINE_8);
		circle(sign, Point(80, 80), 58, red, FILLED, LINE_AA);
		circle(sign, Point(80, 80), 47, white, FILLED, LINE_AA);
		putText(sign, "50", Point(48, 96), FONT_HERSHEY_DUPLEX, 1.35, dark, 3, LINE_AA);
	} else if (shape == SHAPE_TRIANGLE) {
		vector<Point> outer = { Point(80, 16), Point(19, 137), Point(141, 137) };
		vector<Point> inner = { Point(80, 31), Point(36, 125), Point(124, 125) };
		fillConvexPoly(mask, outer, Scalar(255), LINE_8);
		fillConvexPoly(sign, outer, dark, LINE_AA);
		fillConvexPoly(sign, inner, yellow, LINE_AA);
		line(sign, Point(79, 61), Point(79, 100), dark, 6, LINE_AA);
		circle(sign, Point(79, 113), 4, dark, FILLED, LINE_AA);
	} else if (shape == SHAPE_RECTANGLE) {
		RotatedRect rr(Point2f(80, 80), Size2f(112, 92), 0.f);
		Point2f fp[4]; rr.points(fp);
		vector<Point> outer;
		for (int i = 0; i < 4; i++) outer.push_back(fp[i]);
		fillConvexPoly(mask, outer, Scalar(255), LINE_8);
		fillConvexPoly(sign, outer, blue, LINE_AA);
		arrowedLine(sign, Point(45, 82), Point(114, 82), white, 10, LINE_AA, 0, 0.32);
	} else {
		vector<Point> oct;
		for (int i = 0; i < 8; i++) {
			double a = CV_PI / 8.0 + i * CV_PI / 4.0;
			oct.push_back(Point(cvRound(80 + 60 * cos(a)), cvRound(80 + 60 * sin(a))));
		}
		fillConvexPoly(mask, oct, Scalar(255), LINE_8);
		fillConvexPoly(sign, oct, red, LINE_AA);
		putText(sign, "STOP", Point(37, 89), FONT_HERSHEY_DUPLEX, 0.83, white, 3, LINE_AA);
	}
}

static void compositePerspective(Mat &background, Mat &truth, SignShape shape,
	int sample, RNG &rng) {
	Mat sign, sourceMask;
	drawSyntheticSign(shape, sign, sourceMask);

	double halfW = rng.uniform(52.0, 76.0);
	double halfH = halfW * rng.uniform(0.72, 1.05);
	double angle = rng.uniform(-24.0, 24.0) * CV_PI / 180.0;
	Point2f centre((float)rng.uniform(82, 175), (float)rng.uniform(78, 178));
	vector<Point2f> src = { Point2f(12, 12), Point2f(148, 12),
		Point2f(148, 148), Point2f(12, 148) };
	vector<Point2f> dst(4);
	const double sx[4] = { -1, 1, 1, -1 }, sy[4] = { -1, -1, 1, 1 };
	for (int i = 0; i < 4; i++) {
		double x = sx[i] * halfW;
		double y = sy[i] * halfH;
		double rx = x * cos(angle) - y * sin(angle);
		double ry = x * sin(angle) + y * cos(angle);
		dst[i] = Point2f((float)(centre.x + rx + rng.uniform(-7.0, 7.0)),
			(float)(centre.y + ry + rng.uniform(-7.0, 7.0)));
	}
	Mat H = getPerspectiveTransform(src, dst), warpedSign, warpedMask;
	warpPerspective(sign, warpedSign, H, background.size(), INTER_LINEAR,
		BORDER_CONSTANT, Scalar());
	warpPerspective(sourceMask, warpedMask, H, background.size(), INTER_NEAREST,
		BORDER_CONSTANT, Scalar());
	warpedSign.copyTo(background, warpedMask);
	truth = warpedMask;

	// Photometric challenges affect the observed image but never the exact mask.
	if (sample % 4 == 0) {
		Mat shade(background.size(), background.type(), Scalar(35, 35, 35));
		vector<Point> shadow = { Point(0, rng.uniform(70, 150)), Point(255, rng.uniform(125, 215)),
			Point(255, 255), Point(0, 255) };
		Mat sm = Mat::zeros(background.size(), CV_8U);
		fillConvexPoly(sm, shadow, Scalar(145), LINE_8);
		addWeighted(background, 0.68, shade, 0.32, 0.0, shade);
		shade.copyTo(background, sm);
	}
	if (sample % 5 == 0) GaussianBlur(background, background, Size(5, 5), 1.15);
	if (sample % 3 == 0) {
		Mat noise(background.size(), CV_16SC3);
		rng.fill(noise, RNG::NORMAL, Scalar::all(0), Scalar::all(9));
		Mat temp;
		background.convertTo(temp, CV_16SC3);
		add(temp, noise, temp);
		temp.convertTo(background, CV_8UC3);
	}
	if (sample % 7 == 0) {
		vector<uchar> bytes;
		imencode(".jpg", background, bytes, { IMWRITE_JPEG_QUALITY, 52 });
		background = imdecode(bytes, IMREAD_COLOR);
	}
}

bool generateSyntheticBackgroundDataset(const string &root, int imageCount) {
	if (imageCount < 4) imageCount = 4;
	const string imagesDir = root + "/images";
	const string masksDir = root + "/masks";
	utils::fs::createDirectories(imagesDir);
	utils::fs::createDirectories(masksDir);
	ofstream labels((root + "/labels.txt").c_str());
	ofstream metadata((root + "/metadata.csv").c_str());
	if (!labels.is_open() || !metadata.is_open()) return false;
	metadata << "file,shape,background,seed\n";
	const SignShape shapes[4] = { SHAPE_CIRCLE, SHAPE_TRIANGLE,
		SHAPE_RECTANGLE, SHAPE_OCTAGON };
	RNG rng(0x51A9D42u);
	for (int i = 0; i < imageCount; i++) {
		SignShape shape = shapes[i % 4];
		int scene = (i / 4) % 8;
		Mat image, mask;
		drawBackground(image, scene, rng);
		compositePerspective(image, mask, shape, i, rng);
		char name[64];
		sprintf_s(name, "synthetic_%03d.png", i);
		if (!imwrite(imagesDir + "/" + name, image) ||
			!imwrite(masksDir + "/" + name, mask)) return false;
		labels << name << ' ' << shapeName(shape) << '\n';
		metadata << name << ',' << shapeName(shape) << ',' << scene << ",0x51A9D42\n";
	}
	cout << "Generated " << imageCount << " deterministic background-stress images in \""
		<< root << "\".\n";
	return true;
}

static string fileNameOnly(const string &path) {
	size_t pos = path.find_last_of("/\\");
	return pos == string::npos ? path : path.substr(pos + 1);
}

static map<string, SignShape> readShapeLabels(const string &path) {
	map<string, SignShape> labels;
	ifstream in(path.c_str());
	string name, shape;
	while (in >> name >> shape) {
		if (shape == "CIRCLE") labels[name] = SHAPE_CIRCLE;
		else if (shape == "TRIANGLE") labels[name] = SHAPE_TRIANGLE;
		else if (shape == "RECTANGLE") labels[name] = SHAPE_RECTANGLE;
		else if (shape == "OCTAGON") labels[name] = SHAPE_OCTAGON;
	}
	return labels;
}

static double percentile(vector<double> values, double q) {
	if (values.empty()) return 0.0;
	sort(values.begin(), values.end());
	double position = clamp(q, 0.0, 1.0) * (values.size() - 1);
	size_t lower = (size_t)floor(position), upper = (size_t)ceil(position);
	if (lower == upper) return values[lower];
	double fraction = position - lower;
	return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

int evaluateSegmentationDataset(const string &root, const ShapeParams &params,
	bool savePredictions) {
	vector<string> files;
	try { glob(root + "/images/*.png", files, false); }
	catch (const cv::Exception &) { files.clear(); }
	if (files.empty()) {
		cerr << "No PNG evaluation images found under \"" << root << "/images\".\n";
		return -1;
	}
	const string predictionDir = root + "/predictions";
	if (savePredictions) utils::fs::createDirectories(predictionDir);
	ofstream csv((root + "/evaluation.csv").c_str());
	csv << "file,expected_shape,detected_shape,proposal_count,selected_source,"
		"selection_score,selected_area_ratio,latency_ms,iou,dice,precision,recall,boundary_iou\n";
	map<string, SignShape> labels = readShapeLabels(root + "/labels.txt");
	MaskMetrics total;
	MaskMetrics byShape[5];
	int byShapeCount[5] = { 0, 0, 0, 0, 0 };
	vector<double> iouValues, boundaryValues;
	double totalLatencyMs = 0.0;
	int evaluated = 0, shapeScored = 0, shapeCorrect = 0, strongMasks = 0;
	int zeroOverlap = 0, weakMasks = 0;
	cout << "\nPixel-level segmentation evaluation: " << root << "\n";
	for (size_t i = 0; i < files.size(); i++) {
		string name = fileNameOnly(files[i]);
		Mat image = imread(files[i]), truth = imread(root + "/masks/" + name, IMREAD_GRAYSCALE);
		if (image.empty() || truth.empty() || image.size() != truth.size()) {
			cerr << "  skipped " << name << " (missing/incompatible image or mask)\n";
			continue;
		}
		int64 tickStart = getTickCount();
		vector<Mat> masks(3);
		buildColorMasks(image, params, masks[0], masks[1], masks[2]);
		vector<ShapeInfo> candidates = detectSignShapesAdvanced(image, masks, params);
		int best = selectBestSign(candidates, params);
		SignShape found = SHAPE_UNKNOWN;
		Mat prediction = Mat::zeros(image.size(), CV_8U);
		char selectedSource = '-';
		double selectedScore = 0.0, selectedAreaRatio = 0.0;
		if (best >= 0) {
			found = candidates[best].shape;
			prediction = refineShapeMask(image, candidates[best], masks, params);
			selectedSource = candidates[best].proposalSource == SOURCE_EDGE ? 'E' :
				(candidates[best].proposalSource == SOURCE_HOUGH ? 'H' : 'C');
			selectedScore = candidates[best].selectionScore;
			selectedAreaRatio = candidates[best].hullArea /
				max(1.0, (double)image.cols * image.rows);
		}
		double latencyMs = 1000.0 * (getTickCount() - tickStart) / getTickFrequency();
		MaskMetrics m = evaluateBinaryMask(prediction, truth);
		total.iou += m.iou; total.dice += m.dice; total.precision += m.precision;
		total.recall += m.recall; total.boundaryIoU += m.boundaryIoU;
		evaluated++;
		totalLatencyMs += latencyMs;
		iouValues.push_back(m.iou);
		boundaryValues.push_back(m.boundaryIoU);
		if (m.iou >= 0.80 && m.boundaryIoU >= 0.55) strongMasks++;
		if (m.iou < 0.50) weakMasks++;
		if (m.iou == 0.0) zeroOverlap++;
		SignShape expected = SHAPE_UNKNOWN;
		map<string, SignShape>::const_iterator it = labels.find(name);
		if (it != labels.end()) {
			expected = it->second; shapeScored++;
			if (found == expected) shapeCorrect++;
			byShape[expected].iou += m.iou;
			byShape[expected].dice += m.dice;
			byShape[expected].precision += m.precision;
			byShape[expected].recall += m.recall;
			byShape[expected].boundaryIoU += m.boundaryIoU;
			byShapeCount[expected]++;
		}
		csv << name << ',' << shapeName(expected) << ',' << shapeName(found) << ','
			<< candidates.size() << ',' << selectedSource << fixed << setprecision(5)
			<< ',' << selectedScore << ',' << selectedAreaRatio << ',' << latencyMs
			<< ',' << m.iou << ',' << m.dice << ','
			<< m.precision << ',' << m.recall << ',' << m.boundaryIoU << '\n';
		cout << "  " << setw(17) << left << name << " " << setw(9)
			<< shapeName(found) << right << " IoU " << fixed << setprecision(3)
			<< m.iou << "  Dice " << m.dice << "  Boundary-IoU " << m.boundaryIoU << '\n';
		if (savePredictions) imwrite(predictionDir + "/" + name, prediction);
	}
	if (evaluated == 0) return -2;
	double n = (double)evaluated;
	ostringstream report;
	report << "================ pixel-mask summary =====================\n"
		<< "images evaluated       : " << evaluated << '\n'
		<< "mean IoU               : " << fixed << setprecision(3) << total.iou / n << '\n'
		<< "mean Dice              : " << total.dice / n << '\n'
		<< "mean precision         : " << total.precision / n << '\n'
		<< "mean recall            : " << total.recall / n << '\n'
		<< "mean Boundary-IoU      : " << total.boundaryIoU / n << '\n'
		<< "median IoU             : " << percentile(iouValues, 0.50) << '\n'
		<< "10th-percentile IoU    : " << percentile(iouValues, 0.10) << '\n'
		<< "median Boundary-IoU    : " << percentile(boundaryValues, 0.50) << '\n'
		<< "strong masks           : " << strongMasks << '/' << evaluated
		<< "  (IoU >= .80 and Boundary-IoU >= .55)\n"
		<< "weak masks             : " << weakMasks << '/' << evaluated << "  (IoU < .50)\n"
		<< "zero-overlap masks     : " << zeroOverlap << '/' << evaluated << '\n'
		<< "mean pipeline latency  : " << setprecision(2) << totalLatencyMs / n << " ms/image\n"
		<< "measured throughput    : " << setprecision(1)
		<< (totalLatencyMs > 0.0 ? 1000.0 * n / totalLatencyMs : 0.0) << " images/s\n";
	if (shapeScored)
		report << "shape accuracy          : " << shapeCorrect << '/' << shapeScored
			<< "  (" << setprecision(1) << 100.0 * shapeCorrect / shapeScored << "%)\n";
	report << "\nper-shape segmentation (mean IoU / Boundary-IoU)\n";
	const SignShape order[4] = { SHAPE_CIRCLE, SHAPE_TRIANGLE,
		SHAPE_RECTANGLE, SHAPE_OCTAGON };
	for (int i = 0; i < 4; i++) {
		int count = byShapeCount[order[i]];
		report << "  " << setw(9) << left << shapeName(order[i]) << right << " : ";
		if (count > 0)
			report << fixed << setprecision(3) << byShape[order[i]].iou / count
				<< " / " << byShape[order[i]].boundaryIoU / count
				<< "  (n=" << count << ")\n";
		else report << "n/a\n";
	}
	report << "=========================================================\n";
	cout << '\n' << report.str();
	ofstream summary((root + "/evaluation_summary.txt").c_str());
	if (summary.is_open()) summary << report.str();
	return 0;
}

static bool hasSuffix(const string &value, const string &suffix) {
	return value.size() >= suffix.size() &&
		value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static string lowerCase(string value) {
	transform(value.begin(), value.end(), value.begin(),
		[](unsigned char ch) { return (char)tolower(ch); });
	return value;
}

static string fileStemOnly(const string &path) {
	string name = fileNameOnly(path);
	size_t dot = name.find_last_of('.');
	return dot == string::npos ? name : name.substr(0, dot);
}

static void centredSquareCrop(const Mat &image, const Mat &mask,
	const Rect &object, double contextScale, Mat &imageCrop, Mat &maskCrop,
	Rect *objectInCrop = NULL) {
	int side = cvRound(max(object.width, object.height) * contextScale);
	side = max(side, max(object.width, object.height));
	int centreX = object.x + object.width / 2;
	int centreY = object.y + object.height / 2;
	int x = centreX - side / 2, y = centreY - side / 2;
	int left = max(0, -x), top = max(0, -y);
	int right = max(0, x + side - image.cols);
	int bottom = max(0, y + side - image.rows);
	Mat paddedImage, paddedMask;
	copyMakeBorder(image, paddedImage, top, bottom, left, right,
		BORDER_REPLICATE);
	copyMakeBorder(mask, paddedMask, top, bottom, left, right,
		BORDER_CONSTANT, Scalar(0));
	Rect cropRect(x + left, y + top, side, side);
	imageCrop = paddedImage(cropRect).clone();
	maskCrop = paddedMask(cropRect).clone();
	if (objectInCrop)
		*objectInCrop = Rect(object.x + left - cropRect.x,
			object.y + top - cropRect.y, object.width, object.height);
}

static vector<Point> sampledRectangle(const Rect &r) {
	int x0 = r.x, y0 = r.y, x1 = r.x + r.width - 1,
		y1 = r.y + r.height - 1;
	int xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
	return { Point(x0, y0), Point(xm, y0), Point(x1, y0), Point(x1, ym),
		Point(x1, y1), Point(xm, y1), Point(x0, y1), Point(x0, ym) };
}

static uint64 stableInstanceSeed(const string &stem, int component) {
	// FNV-1a gives the evaluation crop a repeatable OpenCV RNG state.  GrabCut's
	// internal GMM initialisation consumes cv::theRNG(); without a per-instance
	// seed, an identical sign can produce a different mask when the surrounding
	// dataset split or traversal order changes.
	uint64 value = UINT64_C(1469598103934665603);
	for (size_t i = 0; i < stem.size(); i++) {
		value ^= (unsigned char)stem[i];
		value *= UINT64_C(1099511628211);
	}
	value ^= (uint64)(unsigned int)component;
	value *= UINT64_C(1099511628211);
	return value == 0 ? UINT64_C(0x9e3779b97f4a7c15) : value;
}

static Mat camVidDiagnostic(const Mat &image, const Mat &truth,
	const Mat &prediction, const MaskMetrics &metrics) {
	Mat truthView = image.clone(), predictionView = image.clone(), errorView = image.clone();
	Mat truthBinary = binary8(truth), predictionBinary = binary8(prediction);
	Mat truthTint(image.size(), image.type(), Scalar(35, 210, 35));
	Mat predictionTint(image.size(), image.type(), Scalar(35, 35, 225));
	Mat tinted;
	addWeighted(image, 0.48, truthTint, 0.52, 0.0, tinted);
	tinted.copyTo(truthView, truthBinary);
	addWeighted(image, 0.48, predictionTint, 0.52, 0.0, tinted);
	tinted.copyTo(predictionView, predictionBinary);

	Mat overlap, falseNegative, falsePositive;
	bitwise_and(truthBinary, predictionBinary, overlap);
	bitwise_and(truthBinary, predictionBinary == 0, falseNegative);
	bitwise_and(predictionBinary, truthBinary == 0, falsePositive);
	errorView.setTo(Scalar(0, 210, 255), overlap);       // yellow: correct
	errorView.setTo(Scalar(0, 190, 0), falseNegative);   // green: missed truth
	errorView.setTo(Scalar(0, 0, 230), falsePositive);   // red: extra prediction

	auto caption = [](Mat &panel, const string &text) {
		rectangle(panel, Rect(0, 0, panel.cols, 25), Scalar(0, 0, 0), FILLED);
		putText(panel, text, Point(6, 18), FONT_HERSHEY_SIMPLEX, 0.48,
			Scalar(255, 255, 255), 1, LINE_AA);
	};
	Mat original = image.clone();
	caption(original, "Original real-background crop");
	caption(truthView, "CamVid SignSymbol ground truth");
	caption(predictionView, "Pipeline prediction");
	ostringstream score;
	score << fixed << setprecision(3) << "Error: IoU " << metrics.iou
		<< "  B-IoU " << metrics.boundaryIoU;
	caption(errorView, score.str());
	Mat top, bottom, diagnostic;
	hconcat(original, truthView, top);
	hconcat(predictionView, errorView, bottom);
	vconcat(top, bottom, diagnostic);
	return diagnostic;
}

int evaluateCamVidDataset(const string &datasetRoot, const string &outputRoot,
	const ShapeParams &params, bool savePredictions, const string &split,
	double contextScale, const string &originalLabelRoot, bool useBoxPrompt) {
	string requestedSplit = lowerCase(split);
	if (requestedSplit != "all" && requestedSplit != "train" &&
		requestedSplit != "val" && requestedSplit != "test") {
		cerr << "Invalid CamVid split \"" << split
			<< "\". Use all, train, val, or test.\n";
		return -1;
	}
	if (contextScale < 1.05 || contextScale > 5.0) {
		cerr << "CamVid context scale must be in [1.05, 5.0].\n";
		return -1;
	}
	vector<string> pngFiles;
	try { glob(datasetRoot + "/*.png", pngFiles, true); }
	catch (const cv::Exception &) { pngFiles.clear(); }
	if (pngFiles.empty()) {
		cerr << "No PNG files found below CamVid root \"" << datasetRoot << "\".\n";
		return -1;
	}

	vector<string> originalLabelFiles;
	if (!originalLabelRoot.empty()) {
		try { glob(originalLabelRoot + "/*.png", originalLabelFiles, true); }
		catch (const cv::Exception &) { originalLabelFiles.clear(); }
		if (originalLabelFiles.empty()) {
			cerr << "No original CamVid label PNGs found below \""
				<< originalLabelRoot << "\".\n";
			return -2;
		}
	}
	bool useOriginal32Labels = !originalLabelFiles.empty();
	map<string, string> sourceImages;
	vector<string> labelFiles;
	for (size_t i = 0; i < pngFiles.size(); i++) {
		string name = fileNameOnly(pngFiles[i]);
		string lowerPath = lowerCase(pngFiles[i]);
		bool annotationPath = lowerPath.find("annot") != string::npos;
		bool splitMatches = requestedSplit == "all" ||
			lowerPath.find("\\" + requestedSplit +
				(annotationPath ? "annot\\" : "\\")) != string::npos ||
			lowerPath.find("/" + requestedSplit +
				(annotationPath ? "annot/" : "/")) != string::npos;
		if (!splitMatches) continue;
		// The downloaded CamVid mirror provides one indexed class-ID mask for
		// every image and optional visualisations ending in _c / _c_c.  Use the
		// indexed fallback mask: class 6 is the reduced CamVid11 SignSymbol
		// category.  It is broader than the original CamVid32 physical-sign
		// target, so corrected evaluations should pass --camvid-label-root.
		if (!useOriginal32Labels && annotationPath && !hasSuffix(name, "_c.png") &&
			!hasSuffix(name, "_c_c.png"))
			labelFiles.push_back(pngFiles[i]);
		else if (!annotationPath)
			sourceImages[fileStemOnly(name)] = pngFiles[i];
	}
	if (useOriginal32Labels) {
		for (size_t i = 0; i < originalLabelFiles.size(); i++)
			if (hasSuffix(fileNameOnly(originalLabelFiles[i]), "_L.png"))
				labelFiles.push_back(originalLabelFiles[i]);
	}
	if (labelFiles.empty() || sourceImages.empty()) {
		cerr << "CamVid images/labels were not both found below \"" << datasetRoot
			<< "\". Expected train/val/test PNGs and matching *annot class masks.\n";
		return -2;
	}

	const string predictionDir = outputRoot + "/predictions";
	const string truthDir = outputRoot + "/truth";
	const string diagnosticDir = outputRoot + "/diagnostics";
	utils::fs::createDirectories(outputRoot);
	if (savePredictions) {
		utils::fs::createDirectories(predictionDir);
		utils::fs::createDirectories(truthDir);
		utils::fs::createDirectories(diagnosticDir);
	}
	ofstream csv((outputRoot + "/evaluation.csv").c_str());
	if (!csv.is_open()) {
		cerr << "Could not create CamVid report below \"" << outputRoot << "\".\n";
		return -3;
	}
	csv << "sample,source_image,component,original_x,original_y,original_width,"
		"original_height,original_pixels,detected_shape,proposal_count,selected_source,"
		"selection_score,latency_ms,iou,dice,precision,recall,boundary_iou\n";

	// CamVid's original 32-class palette keeps SignSymbol separate from
	// TrafficLight.  It uses RGB (192,128,128), hence BGR (128,128,192) in
	// OpenCV.  The smaller mirror's CamVid11 indexed masks merge more categories;
	// class ID 6 remains available only as a documented fallback.
	const int signSymbolClassId = 6;
	const Scalar signSymbolBgr(128, 128, 192);
	const int normalisedSize = 256;
	const int minimumSide = 10;
	const int minimumPixels = 30;
	ShapeParams externalParams = params;
	// Real CamVid signs are often neutral white/grey plates whose only coloured
	// pixels belong to an inner pictogram.  The school profile intentionally
	// requires traffic-sign colour, but that gate removes the actual outer plate
	// here.  This evaluation profile admits independently strong edge proposals,
	// gives the centred candidate contract substantially more weight, and avoids
	// ranking a large distractor above the prompted sign merely due to area.
	externalParams.minArea = 40.0;
	externalParams.minAreaRatio = 0.002;
	externalParams.maxAspect = 4.5;
	externalParams.minColorCoverage = 0.0;
	externalParams.minEdgeSupport = 0.24;
	externalParams.selectionSizeWeight = 0.60;
	externalParams.centerPriorStrength = 0.55;

	MaskMetrics total;
	vector<double> iouValues, boundaryValues;
	double totalLatencyMs = 0.0;
	int pairedFrames = 0, incompatibleFrames = 0, labelledFrames = 0;
	int connectedRegions = 0, skippedTiny = 0, evaluated = 0;
	int detected = 0, strongMasks = 0, weakMasks = 0, zeroOverlap = 0;
	cout << "\nCamVid real-background pixel-mask evaluation\n"
		<< "  dataset : " << datasetRoot << '\n'
		<< "  outputs : " << outputRoot << '\n'
		<< "  split   : " << requestedSplit << '\n'
		<< "  labels  : " << (useOriginal32Labels ?
			"original CamVid32 SignSymbol" : "CamVid11 merged SignSymbol") << '\n'
		<< "  prompt  : " << (useBoxPrompt ?
			"ground-truth box (segmentation-only protocol)" : "none") << '\n'
		<< "  protocol: connected SignSymbol regions >= " << minimumSide
		<< " px per side and >= " << minimumPixels << " pixels; "
		<< contextScale << "x square context resized to " << normalisedSize << "x"
		<< normalisedSize << "\n\n";

	for (size_t labelIndex = 0; labelIndex < labelFiles.size(); labelIndex++) {
		string labelStem = fileStemOnly(labelFiles[labelIndex]);
		if (hasSuffix(labelStem, "_L")) labelStem.resize(labelStem.size() - 2);
		map<string, string>::const_iterator source = sourceImages.find(labelStem);
		if (source == sourceImages.end()) continue;
		Mat image = imread(source->second, IMREAD_COLOR);
		Mat label = imread(labelFiles[labelIndex],
			useOriginal32Labels ? IMREAD_COLOR : IMREAD_GRAYSCALE);
		if (image.empty() || label.empty()) {
			incompatibleFrames++;
			continue;
		}
		if (label.size() != image.size())
			resize(label, label, image.size(), 0.0, 0.0, INTER_NEAREST);
		pairedFrames++;
		Mat signMask;
		if (useOriginal32Labels)
			inRange(label, signSymbolBgr, signSymbolBgr, signMask);
		else
			compare(label, signSymbolClassId, signMask, CMP_EQ);
		if (countNonZero(signMask) == 0) continue;
		labelledFrames++;

		Mat componentLabels, stats, centroids;
		int componentCount = connectedComponentsWithStats(signMask,
			componentLabels, stats, centroids, 8, CV_32S);
		for (int component = 1; component < componentCount; component++) {
			connectedRegions++;
			int originalPixels = stats.at<int>(component, CC_STAT_AREA);
			Rect object(stats.at<int>(component, CC_STAT_LEFT),
				stats.at<int>(component, CC_STAT_TOP),
				stats.at<int>(component, CC_STAT_WIDTH),
				stats.at<int>(component, CC_STAT_HEIGHT));
			if (object.width < minimumSide || object.height < minimumSide ||
				originalPixels < minimumPixels) {
				skippedTiny++;
				continue;
			}

			Mat instanceMask;
			compare(componentLabels, component, instanceMask, CMP_EQ);
			Mat sourceCrop, sourceTruth, crop, truth;
			Rect objectInSourceCrop;
			centredSquareCrop(image, instanceMask, object, contextScale,
				sourceCrop, sourceTruth, &objectInSourceCrop);
			resize(sourceCrop, crop, Size(normalisedSize, normalisedSize),
				0.0, 0.0, INTER_AREA);
			resize(sourceTruth, truth, Size(normalisedSize, normalisedSize),
				0.0, 0.0, INTER_NEAREST);
			double promptScale = (double)normalisedSize / sourceCrop.cols;
			Rect promptBox(cvRound(objectInSourceCrop.x * promptScale),
				cvRound(objectInSourceCrop.y * promptScale),
				max(4, cvRound(objectInSourceCrop.width * promptScale)),
				max(4, cvRound(objectInSourceCrop.height * promptScale)));
			promptBox &= Rect(0, 0, normalisedSize, normalisedSize);
			if (useBoxPrompt && promptBox.width >= 12 && promptBox.height >= 12) {
				// Semantic boxes include a one-pixel source-resolution annotation
				// margin.  At 256x256 this becomes a visibly oversized prior.  A
				// fixed 5% linear calibration was selected on the validation split;
				// it is frozen before the test split and never reads truth pixels.
				int calibratedWidth = max(4, cvRound(0.95 * promptBox.width));
				int calibratedHeight = max(4, cvRound(0.95 * promptBox.height));
				int calibratedX = promptBox.x + (promptBox.width - calibratedWidth) / 2;
				int calibratedY = promptBox.y + (promptBox.height - calibratedHeight) / 2;
				promptBox = Rect(calibratedX, calibratedY,
					calibratedWidth, calibratedHeight) &
					Rect(0, 0, normalisedSize, normalisedSize);
			}

			uint64 previousRngState = theRNG().state;
			theRNG().state = stableInstanceSeed(labelStem, component);
			int64 tickStart = getTickCount();
			vector<Mat> colourMasks(3);
			buildColorMasks(crop, externalParams, colourMasks[0], colourMasks[1],
				colourMasks[2]);
			vector<ShapeInfo> candidates = detectSignShapesAdvanced(crop,
				colourMasks, externalParams);
			int best = selectBestSign(candidates, externalParams);
			Mat prediction = Mat::zeros(crop.size(), CV_8U);
			SignShape found = SHAPE_UNKNOWN;
			char selectedSource = '-';
			double selectedScore = 0.0;
			ShapeInfo selectedInfo;
			bool haveSelection = false;
			if (best >= 0) {
				selectedInfo = candidates[best];
				haveSelection = true;
				selectedSource = selectedInfo.proposalSource == SOURCE_EDGE ? 'E' :
					(selectedInfo.proposalSource == SOURCE_HOUGH ? 'H' : 'C');
				selectedScore = selectedInfo.selectionScore;
			}
			if (useBoxPrompt && promptBox.area() > 0) {
				ShapeInfo boxInfo;
				vector<Point> boxContour = sampledRectangle(promptBox);
				if (classifyContour(boxContour, crop.size(), externalParams, boxInfo)) {
					// Retain a detected circular/triangular/octagonal outline only when
					// it spans nearly the full prompt in both axes.  This preserves a
					// direct round sign but rejects a small pictogram printed inside a
					// larger neutral rectangular physical plate.
					bool detectedOuterShape = false;
					if (haveSelection && selectedInfo.shape != SHAPE_RECTANGLE) {
						double spanX = (double)selectedInfo.bbox.width /
							max(1, promptBox.width);
						double spanY = (double)selectedInfo.bbox.height /
							max(1, promptBox.height);
						Point2f pc(promptBox.x + 0.5f * promptBox.width,
							promptBox.y + 0.5f * promptBox.height);
						double dx = abs(selectedInfo.box.center.x - pc.x) /
							max(1.0, (double)promptBox.width);
						double dy = abs(selectedInfo.box.center.y - pc.y) /
							max(1.0, (double)promptBox.height);
						detectedOuterShape = spanX >= 0.78 && spanY >= 0.78 &&
							dx <= 0.16 && dy <= 0.16;
					}
					if (!detectedOuterShape) {
						selectedInfo = boxInfo;
						haveSelection = true;
						selectedSource = 'B';
						selectedScore = 1.0;
					}
				}
			}
			if (haveSelection) {
				found = selectedInfo.shape;
				prediction = refineShapeMask(crop, selectedInfo, colourMasks,
					externalParams);
				detected++;
			}
			theRNG().state = previousRngState;
			double latencyMs = 1000.0 * (getTickCount() - tickStart) /
				getTickFrequency();
			MaskMetrics metrics = evaluateBinaryMask(prediction, truth);
			total.iou += metrics.iou;
			total.dice += metrics.dice;
			total.precision += metrics.precision;
			total.recall += metrics.recall;
			total.boundaryIoU += metrics.boundaryIoU;
			totalLatencyMs += latencyMs;
			iouValues.push_back(metrics.iou);
			boundaryValues.push_back(metrics.boundaryIoU);
			if (metrics.iou >= 0.80 && metrics.boundaryIoU >= 0.55) strongMasks++;
			if (metrics.iou < 0.50) weakMasks++;
			if (metrics.iou == 0.0) zeroOverlap++;

			ostringstream sampleName;
			sampleName << labelStem << "_sign" << setw(2) << setfill('0') << component;
			csv << sampleName.str() << ',' << fileNameOnly(source->second) << ','
				<< component << ',' << object.x << ',' << object.y << ','
				<< object.width << ',' << object.height << ',' << originalPixels << ','
				<< shapeName(found) << ',' << candidates.size() << ',' << selectedSource
				<< fixed << setprecision(5) << ',' << selectedScore << ',' << latencyMs
				<< ',' << metrics.iou << ',' << metrics.dice << ',' << metrics.precision
				<< ',' << metrics.recall << ',' << metrics.boundaryIoU << '\n';
			if (savePredictions) {
				imwrite(predictionDir + "/" + sampleName.str() + ".png", prediction);
				imwrite(truthDir + "/" + sampleName.str() + ".png", truth);
				imwrite(diagnosticDir + "/" + sampleName.str() + ".png",
					camVidDiagnostic(crop, truth, prediction, metrics));
			}
			evaluated++;
			cout << "  [" << evaluated << "] " << left << setw(26) << sampleName.str()
				<< right << " " << setw(9) << shapeName(found) << " IoU " << fixed
				<< setprecision(3) << metrics.iou << "  Boundary-IoU "
				<< metrics.boundaryIoU << '\n';
		}
	}
	if (evaluated == 0) {
		cerr << "No eligible CamVid SignSymbol regions were found. Check that the "
			<< "original 32-class palette labels were extracted.\n";
		return -4;
	}

	double n = (double)evaluated;
	ostringstream report;
	report << "================ CamVid exact-mask summary ==============\n"
		<< "dataset type           : real dashboard street scenes\n"
		<< "evaluated split        : " << requestedSplit << '\n'
		<< "context scale          : " << fixed << setprecision(2)
		<< contextScale << "x target max side\n"
		<< "box prompt             : " << (useBoxPrompt ? "yes" : "no") << '\n'
		<< "ground truth class     : " << (useOriginal32Labels
			? "CamVid32 SignSymbol RGB(192,128,128), TrafficLight excluded\n"
			: "CamVid11 merged SignSymbol class ID 6\n")
		<< "paired labelled files : " << pairedFrames << '\n'
		<< "files containing signs: " << labelledFrames << '\n'
		<< "connected sign regions: " << connectedRegions << '\n'
		<< "tiny regions excluded : " << skippedTiny
		<< "  (<10 px side or <30 source pixels)\n"
		<< "regions evaluated      : " << evaluated << '\n'
		<< "non-empty predictions : " << detected << '/' << evaluated << '\n'
		<< "mean IoU               : " << fixed << setprecision(3) << total.iou / n << '\n'
		<< "mean Dice              : " << total.dice / n << '\n'
		<< "mean precision         : " << total.precision / n << '\n'
		<< "mean recall            : " << total.recall / n << '\n'
		<< "mean Boundary-IoU      : " << total.boundaryIoU / n << '\n'
		<< "median IoU             : " << percentile(iouValues, 0.50) << '\n'
		<< "10th-percentile IoU    : " << percentile(iouValues, 0.10) << '\n'
		<< "median Boundary-IoU    : " << percentile(boundaryValues, 0.50) << '\n'
		<< "strong masks           : " << strongMasks << '/' << evaluated
		<< "  (IoU >= .80 and Boundary-IoU >= .55)\n"
		<< "weak masks             : " << weakMasks << '/' << evaluated
		<< "  (IoU < .50)\n"
		<< "zero-overlap masks     : " << zeroOverlap << '/' << evaluated << '\n'
		<< "mean pipeline latency  : " << setprecision(2) << totalLatencyMs / n
		<< " ms/crop\n"
		<< "measured throughput    : " << setprecision(1)
		<< (totalLatencyMs > 0.0 ? 1000.0 * n / totalLatencyMs : 0.0)
		<< " crops/s\n"
		<< "incompatible pairs     : " << incompatibleFrames << '\n'
		<< "=========================================================\n";
	cout << '\n' << report.str();
	ofstream summary((outputRoot + "/evaluation_summary.txt").c_str());
	if (summary.is_open()) summary << report.str();
	return 0;
}
